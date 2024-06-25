import cv2
import os
import torch
from torchvision import transforms
from typing import List, Tuple
from PIL import Image
from ultralytics import YOLO
# Constants and Model Paths
VIDEO_PATH = '/home/raptor1/Desktop/Vessels/videos/passanger_cruise1.mov'

YOLO_MODEL_PATH = '/home/raptor1/Desktop/Vessels/Testing/models/Vessel_det.pt'
IMAGE_CLF_MODEL_PATH = '/home/raptor1/Desktop/Vessels/Testing/models/vessel_vision.pt'

CLASS_NAMES = ['Aircraft_Carrier', 'Container_cargo', 'Destroyer_Frigate_Corverttes',
                   'FACs','Fishing_ship', 'LDP_LHP', 'Oil_Tanker_Bulk_carrier','Passanger_Cruise', 'Sail_ship','Tug_ship']



# Initialize YOLO model
yolo = YOLO(YOLO_MODEL_PATH)
# Load image classification model
clf = torch.load(IMAGE_CLF_MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf.to(device)  # Move the model to the appropriate device
# Open video stream
cap = cv2.VideoCapture(VIDEO_PATH)
# Define the output video writer
output_path = 'output_video3.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (1000, 620))  # Adjust frame size as needed

def process_roi(frame, box, clf, class_names):
    image_transform = transforms.Compose([
    transforms.Resize(224),  # Resize the shorter side to 224 pixels while maintaining aspect ratio
    transforms.CenterCrop((224, 224)),  # Center crop to obtain a 224x224 image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    # Implement your ROI processing logic here
    # This function should take a frame, box, classification model (clf), and class names as parameters
    # Perform classification and drawing operations
    frame2 = frame.copy()
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame.shape[1] - 1, int(x2))
    y2 = min(frame.shape[0] - 1, int(y2))
    roi = frame2[int(y1 - 15):int(y2 + 15), int(x1 - 15):int(x2 + 15)]
    cv2.rectangle(frame, (x1 , y1), (x2, y2), (0, 0, 0), 2)
    try:
        img = Image.fromarray(roi)
    except ValueError as e:
        print(f"Error creating PIL Image: {e}")
        return
    clf.to(device)
    clf.eval()
    with torch.no_grad():
        transformed_frame = image_transform(img).unsqueeze(dim=0).to(device)
        frame_pred = clf(transformed_frame)
    frame_pred_probs = torch.softmax(frame_pred, dim=1)
    frame_pred_label = torch.argmax(frame_pred_probs, dim=1)
    label = f"Pred: {class_names[frame_pred_label]} | Prob: {frame_pred_probs.max():.3f}"
    # cv2.imwrite(f"roi_{label}{x1}.jpg", roi)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    if label:
        tf = max(3 - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 2 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(frame, p1, p2, (0, 0, 0), -1, cv2.LINE_AA)  # filled
        cv2.putText(frame, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2 / 3, (255, 255, 255),
                    thickness=tf, lineType=cv2.LINE_AA)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Resize the frame for display
    frame = cv2.resize(frame, (1000, 620))
    # Object detection with YOLO
    results = yolo.predict(frame, imgsz=1920, conf=0.25, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    for box in boxes:
        process_roi(frame, box, clf, CLASS_NAMES)
    # Write the resized frame to the output video
    output_video.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
output_video.release()
cv2.destroyAllWindows()