import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

CLASSES = ['CLEAN', 'MIXED']
CHANNELS = (32, 64, 128, 256, 512)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Best_models/cnn5_k7_binary_best.pt"
STREAM_URL = "http://10.233.25.98:6767/video" 

def _block(c_in: int, c_out: int, kernel_size: int) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

class Cnn5(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        in_ch = 1 
        layers = []
        for i, out_ch in enumerate(CHANNELS):
            k = 7 if i == 0 else 3
            layers.append(_block(in_ch, out_ch, kernel_size=k))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(CHANNELS[-1], num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

model = Cnn5(num_classes=2).to(DEVICE)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"✅ Success: Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

def prepare_cnn5_frame(frame):
    """
    Returns:
        tensor: Normalized tensor for model inference
        display_img: Grayscale, inverted, resized image for visualization
    """
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize (136x68)
    img_pil = Image.fromarray(gray)
    img_resized = img_pil.resize((136, 68), Image.Resampling.LANCZOS)
    
    # 3. Convert to Tensor & Invert
    # We do this manually on the numpy/PIL level for the display view
    img_np = np.array(img_resized)
    img_inverted = 255 - img_np

    # 4. Final Tensor for Model
    tensor = transforms.ToTensor()(Image.fromarray(img_inverted))
    
    # Scale up the filtered image so it's visible (e.g., 4x)
    #display_view = cv2.resize(img_inverted, (136*4, 68*4), interpolation=cv2.INTER_LANCZOS4)
    display_view =img_inverted  # --- IGNORE --- (Use original resized view instead of inverted for display)
    
    return tensor.unsqueeze(0).to(DEVICE), display_view

# --- 5. MAIN FEED LOOP ---
cap = cv2.VideoCapture(STREAM_URL)

print("\n Starting Real-Time Inference...")
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Lost video feed. Retrying...")
        continue

    # Get both the model input and the visual "filter" view
    input_tensor, filter_view = prepare_cnn5_frame(frame)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = output.argmax(1).item()
        confidence = probs[0][pred_idx].item()

    label = CLASSES[pred_idx]
    
    # UI Styling for Main Window
    text = f"CNN5: {label} ({confidence:.2%})"
    color = (0, 255, 0) if label == 'CLEAN' else (0, 0, 255)
    
    cv2.rectangle(frame, (15, 15), (550, 65), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show Windows
    cv2.imshow("ADL Handwriting Classifier - Main", frame)
    cv2.imshow("Model Input View (Grayscale + Inverted)", filter_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()