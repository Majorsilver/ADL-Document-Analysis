import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup, GreedyLR
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import f1_score
import wandb
from tqdm import tqdm


CLASSES = ['CLEAN', 'CROSS', 'DIAGONAL', 'DOUBLE_LINE', 'SCRATCH', 'SINGLE_LINE', 'WAVE', 'ZIG_ZAG']
BASE_PATH = "/home/jovyan/Original_Dataset"

CONFIG = {
    "lr": 1e-4,          
    "epochs": 100,
    "batch_size": 128,
    "img_size": 128,
    "num_classes": len(CLASSES),
    "patience": 10,
    "min_lr": 1e-6,
}


class CrossOutDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=128):
        self.samples = []
        self.transform = transform
        
        # Loop through our array to find the folders we want
        for idx, class_name in enumerate(CLASSES):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path): continue
            
            for img_name in os.listdir(class_path):
                self.samples.append((os.path.join(class_path, img_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img.thumbnail((128, 128), Image.Resampling.LANCZOS)        
        new_img = Image.new("RGB", (128, 128), (255, 255, 255))
        img_w, img_h = img.size
        offset = ((128 - img_w) // 2, (128 - img_h) // 2)
        new_img.paste(img, offset)
        
        if self.transform:
            new_img = self.transform(new_img)
            
        return new_img, label

# Yolo
class YOLOClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Loads YOLO11 and grabs the first 10 layers (the backbone)
        yolo_backbone = YOLO("yolo11n.pt").model
        self.features = nn.Sequential(*list(yolo_backbone.children())[0][:10])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, CONFIG["num_classes"])
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

# Training and Eval
def run_eval(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            out = model(imgs)
            total_loss += criterion(out, labels).item()
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item() * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, total_loss / len(loader), f1, all_preds, all_labels

def train():
    wandb.init(project="cross-out-detection", name="Yolo11 - Multiclass - FineTuning - No Preprocessing", config=CONFIG) # Setup wandb
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize(mean=[0.06083, 0.06083, 0.06083], 
                             std=[0.19272, 0.19272, 0.19272]) # Calculated with mean_std script
    ])

    train_loader = DataLoader(CrossOutDataset(f"{BASE_PATH}/train/images", transform), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(CrossOutDataset(f"{BASE_PATH}/val/images", transform), batch_size=CONFIG["batch_size"], num_workers=0)
    test_loader = DataLoader(CrossOutDataset(f"{BASE_PATH}/test/images", transform), batch_size=CONFIG["batch_size"], num_workers=0)

    model = YOLOClassifier().cuda()
    # GPU parallization
    #if torch.cuda.device_count() > 1:
    #    print(f"Number of GPUs used for training:{torch.cuda.device_count()} !")
    #    model = nn.DataParallel(model)
    model = model.cuda()

    actual_model = model.module if isinstance(model, nn.DataParallel) else model #Unwrap the model from DataParallel (i)

    optimizer = optim.Adam(actual_model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor=0.5,      
        patience=5,
        min_lr=CONFIG["min_lr"],
        verbose=True
    )

    # Early stopping variables 
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = "Best_models/Yolo11_FineTuning_MultiClass_NoPreprocessing.pth"

    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad()

            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
                        
            train_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # Metrics
        train_acc = (correct / total) * 100
        val_acc, val_loss, val_f1, _, _ = run_eval(model, val_loader)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        # Wandb logging
        wandb.log({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss / len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "f1_score": val_f1
        })

        # Early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f"\nValidation loss improved to {val_loss:.4f}. Saving model.")
        else:
            patience_counter += 1
            print(f"\nNo improvement in validation loss for {patience_counter} epochs.")
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break

    # Load the best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("\nLoaded best model weights for final testing.")

    # Final test
    test_acc, test_loss, test_f1, test_preds, test_labels = run_eval(model, test_loader)

    # Send info to wandb
    wandb.log({
        "test_acc": test_acc,
        "test_loss": test_loss,
        "conf_mat": wandb.plot.confusion_matrix(y_true=test_labels, preds=test_preds, class_names=CLASSES)
    })
    
    wandb.finish()
    
if __name__ == "__main__":
    train()