import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

CLASSES = ['CLEAN', 'CROSS', 'DIAGONAL', 'DOUBLE_LINE', 'SCRATCH', 'SINGLE_LINE', 'WAVE', 'ZIG_ZAG']
BASE_PATH = "Original_Dataset"

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

def calculate_mean_std():
    # IMPORTANT: We only use Resize, ToTensor, and your Lambda flip. 
    # Do NOT include the Normalize step here!
    calc_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x) 
    ])

    train_dataset = CrossOutDataset(f"{BASE_PATH}/train/images", transform=calc_transform)
    
    loader = DataLoader(train_dataset, batch_size=256, num_workers=4, shuffle=False)

    channels_sum, channels_sq_sum, num_batches = 0, 0, 0

    print(f"Calculating Mean and Std across {len(train_dataset)} training images...")
    
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sq_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_sq_sum / num_batches - mean ** 2) ** 0.5

    print("\n--- RESULTS ---")
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")

if __name__ == "__main__":
    calculate_mean_std()