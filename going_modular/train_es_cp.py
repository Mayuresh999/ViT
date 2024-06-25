import torch
import torchvision
import os
from torch import nn
from torchvision import transforms
from transformers.trainer_utils import set_seed
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from going_modular.going_modular import engine

# Define your early stopping criteria
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

if __name__ == '__main__':
# Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    # 2. Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

    # 3. Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Setup directory paths to train and test images
    train_dir = r'E:\windowsps_backup\drone_det\drone_classification\train'
    test_dir = r'E:\windowsps_backup\drone_det\drone_classification\val'

    # 4. Change the classifier head
    class_names = ['Fixed_wing', 'Payload_drones', 'Quadcopters']

    set_seed(123)
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
        # Print a summary using torchinfo (uncomment for actual output)
    summary(model=pretrained_vit, 
            input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )




    # Get automatic transforms from pretrained ViT weights
    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    NUM_WORKERS = os.cpu_count()
    def create_dataloaders(
        train_dir: str, 
        test_dir: str, 
        transform: transforms.Compose, 
        batch_size: int, 
        num_workers: int=NUM_WORKERS):

        # Use ImageFolder to create dataset(s)
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

        # Get class names
        class_names = train_data.classes

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_dataloader, test_dataloader, class_names
    # Define a function for training
    def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience, save_path):
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training loop
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}")

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print("Model saved")

    # Setup dataloaders
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                            test_dir=test_dir,
                                                                                            transform=pretrained_vit_transforms,
                                                                                            batch_size=128)

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define the number of epochs, patience, and save path
    num_epochs = 300
    patience = 50
    save_path = 'vit_image_classification_early_stopping.pt'

    # Train with early stopping
    train_with_early_stopping(pretrained_vit, train_dataloader_pretrained, test_dataloader_pretrained, optimizer, loss_fn, device, num_epochs, patience, save_path)