import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# 1. RE-DEFINE THE MODEL ARCHITECTURE (Must match training exactly)
class YOLOClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        yolo_backbone = YOLO("yolo11n.pt").model
        self.features = nn.Sequential(*list(yolo_backbone.children())[0][:10])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

# 2. SETUP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Best_models/Yolo11_FeatureExtractor_Binary_NoPreprocessing_ReduceLROnPlateau.pth"
# Use the new IP from your hotspot (usually 192.168.43.1 or similar)
STREAM_URL = "http://10.209.15.89:6767/video" 

# Load Model
model = YOLOClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Preprocessing Pipeline (Identical to your Training Transform)
transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=5, translate=(0.05, 0.05), shear=5, fill=0
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: 1.0 - x),
    #transforms.Normalize(mean=[0.0550, 0.0550, 0.0550], 
    #                         std=[0.1759, 0.1759, 0.1759]) # Calculated with mean_std script
])

def prepare_frame(frame):
    """Matches your CrossOutDataset logic: Letterboxing + Center Padding"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    # 128x128 Letterboxing
    img.thumbnail((128, 128), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (128, 128), (255, 255, 255))
    img_w, img_h = img.size
    offset = ((128 - img_w) // 2, (128 - img_h) // 2)
    new_img.paste(img, offset)
    
    return transform(new_img).unsqueeze(0).to(DEVICE)

# 3. THE MAIN LOOP
cap = cv2.VideoCapture(STREAM_URL)

print("Starting Real-Time Detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check if IP Webcam is still running.")
        break

    # Inference
    input_tensor = prepare_frame(frame)
    with torch.no_grad():
        output = model(input_tensor)
        # Convert logit to probability
        prob = torch.sigmoid(output).item()
        
    # Logic: 1 = MIXED, 0 = CLEAN
    label = "MIXED" if prob > 0.5 else "CLEAN"
    confidence = prob if prob > 0.5 else 1 - prob
    
    # UI Overlay
    color = (0, 0, 255) if label == "MIXED" else (0, 255, 0) # BGR
    cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1) # Black background for text
    cv2.putText(frame, f"{label}: {confidence:.2%}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Handwriting Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()