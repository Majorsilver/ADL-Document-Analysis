import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import get_scheduler
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import f1_score
import wandb
from tqdm import tqdm


CLASSES = ['CLEAN', 'MIXED']
BASE_PATH = "Original_Dataset"

CONFIG = {
    "lr": 1e-4,       
    "epochs": 1000,   
    "batch_size": 128,
    "img_size": 128,
    "patience": 100,
    "weight_decay": 1e-4,
    "warmup_ratio": 0.1,
    "min_lr": 1e-6,
    "max_lr": 1e-3,
    "factor": 0.95,
    "warmup": 10,
}


class CrossOutDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=128):
        self.samples = []
        self.transform = transform
        
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
        yolo_backbone = YOLO("yolo11n.pt").model
        self.features = nn.Sequential(*list(yolo_backbone.children())[0][:10])

        for p in self.features.parameters(): p.requires_grad = False 
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

# Training and Eval
def run_eval(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss() 
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            labels = labels.float().unsqueeze(1)
            
            out = model(imgs)
            total_loss += criterion(out, labels).item()
            
            preds = (out > 0.0).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    f1 = f1_score(all_labels, all_preds, average='binary') 
    
    return acc, total_loss / len(loader), f1, all_preds, all_labels

def train():
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), shear=5, fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
    ])

    train_loader = DataLoader(CrossOutDataset(f"{BASE_PATH}/train/images", transform), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(CrossOutDataset(f"{BASE_PATH}/val/images", transform), batch_size=CONFIG["batch_size"], num_workers=0)
    test_loader = DataLoader(CrossOutDataset(f"{BASE_PATH}/test/images", transform), batch_size=CONFIG["batch_size"], num_workers=0)

    schedulers_to_test = [
        "linear", 
        "cosine", 
        "cosine_with_restarts", 
        "polynomial", 
        "constant", 
        "constant_with_warmup",
        "inverse_sqrt",
        "reduce_lr_on_plateau",
        "cosine_with_min_lr",
        "cosine_warmup_with_min_lr",
        "warmup_stable_decay",
        "greedy"
    ]

    for sched_name in schedulers_to_test:
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING WITH SCHEDULER: {sched_name.upper()}")
        print(f"{'='*60}\n")
        
        wandb.init(
            project="cross-out-detection", 
            name=f"Yolo11 - Binary - {sched_name} - FeatureExtraction - Preprocessing - DataAugmentation", 
            config=CONFIG
        )

        model = YOLOClassifier().cuda()
        actual_model = model.module if isinstance(model, nn.DataParallel) else model

        optimizer = optim.Adam(actual_model.head.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        criterion = nn.BCEWithLogitsLoss() 
        
        total_training_steps = CONFIG["epochs"] * len(train_loader)
        warmup_steps = int(CONFIG["warmup_ratio"] * total_training_steps)
        
        # Check if the scheduler needs special metric handling
        is_plateau_scheduler = (sched_name == "reduce_lr_on_plateau")
        
        # Note: If any specific scheduler requires extra arguments (like min_lr), 
        # you can pass a dictionary to `scheduler_specific_kwargs` inside get_scheduler.
        try:
            scheduler = get_scheduler(
                name=sched_name,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps
            )
        except ValueError as e:
            print(f"Skipping {sched_name} due to initialization error (might require newer transformers version or specific kwargs): {e}")
            wandb.finish()
            continue

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = f"Best_models/Yolo11_Binary_{sched_name}_FeatureExtraction_Preprocessing_DataAugmentation.pth" 

        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        for epoch in range(CONFIG["epochs"]):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [{sched_name}]"):
                imgs, labels = imgs.cuda(), labels.cuda()
                labels = labels.float().unsqueeze(1) 

                optimizer.zero_grad()

                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                
                # Update learning rate per batch ONLY if it's not reduce_lr_on_plateau
                if not is_plateau_scheduler:
                    scheduler.step() 
                            
                train_loss += loss.item()
                
                preds = (out > 0.0).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            train_acc = (correct / total)
            val_acc, val_loss, val_f1, _, _ = run_eval(model, val_loader)
            
            # Update learning rate per epoch ONLY if it IS reduce_lr_on_plateau
            if is_plateau_scheduler:
                scheduler.step(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
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

        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"\nLoaded best {sched_name} model weights for final testing.")

        test_acc, test_loss, test_f1, test_preds, test_labels = run_eval(model, test_loader)

        wandb.log({
            "test_acc": test_acc,
            "test_loss": test_loss,
            "conf_mat": wandb.plot.confusion_matrix(y_true=test_labels, preds=test_preds, class_names=CLASSES)
        })
        
        wandb.finish()
    
if __name__ == "__main__":
    train()