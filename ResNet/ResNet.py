import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score
import wandb
from tqdm import tqdm

CLASSES = ['CLEAN', 'CROSS', 'DIAGONAL', 'DOUBLE_LINE', 'SCRATCH', 'SINGLE_LINE', 'WAVE', 'ZIG_ZAG']
BASE_PATH = "/home/jovyan/Dataset"

CONFIG = {
    "lr": 1e-3,
    "epochs": 15,
    "batch_size": 128,
    "img_size": 128,
    "num_classes": len(CLASSES)
}

class CrossOutDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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
        img = Image.open(path).convert("RGB") # YOLO likes 3 channels
        if self.transform:
            img = self.transform(img)
        return img, label

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.
        self.block1 = ResidualBlock()

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
    wandb.init(project="CrossOut_Project", name="ResNet", config=CONFIG) # setup wandb
    
    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x), # Flip white ink on black background
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_loader = DataLoader(CrossOutDataset(f"{BASE_PATH}/train/images", transform), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(CrossOutDataset(f"{BASE_PATH}/val/images", transform), batch_size=CONFIG["batch_size"], num_workers=0)
    test_loader = DataLoader(CrossOutDataset(f"{BASE_PATH}/test/images", transform), batch_size=CONFIG["batch_size"], num_workers=0)

    model = YOLOClassifier().cuda()
    optimizer = optim.Adam(model.head.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

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

        # Wandb logging
        wandb.log({
            "epoch": epoch + 1,
            "lr": CONFIG["lr"],
            "train_loss": train_loss / len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "f1_score": val_f1
        })

    # Final test
    test_acc, test_loss, test_f1, test_preds, test_labels = run_eval(model, test_loader)
    
    wandb.log({
        "test_acc": test_acc,
        "test_loss": test_loss,
        "conf_mat": wandb.plot.confusion_matrix(y_true=test_labels, preds=test_preds, class_names=CLASSES)
    })
    
    wandb.finish()
    
if __name__ == "__main__":
    train()