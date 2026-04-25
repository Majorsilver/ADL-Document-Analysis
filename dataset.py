from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

DATASET_ROOT = Path(__file__).parent / "cross_out_dataset_v2"

MEDIAN_W = 136
MEDIAN_H = 68
FINAL_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FitPadInvert:
    """Invert to white-on-black, downscale (no stretch) to fit target, pad with black."""

    def __init__(self, target_w: int = FINAL_SIZE, target_h: int = FINAL_SIZE):
        self.tw = target_w
        self.th = target_h

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        img = ImageOps.invert(img)

        w, h = img.size
        scale = min(self.tw / w, self.th / h, 1.0)
        if scale < 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            w, h = img.size

        pad_left = (self.tw - w) // 2
        pad_top = (self.th - h) // 2
        pad_right = self.tw - w - pad_left
        pad_bottom = self.th - h - pad_top
        img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
        return img


class ToThreeChannel224:
    """Grayscale PIL → 3x224x224 tensor normalized with ImageNet stats."""

    def __init__(self, size: int = FINAL_SIZE):
        self.size = size
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)  # 1×H×W in [0,1]
        t = TF.resize(t, [self.size, self.size], antialias=True)
        t = t.repeat(3, 1, 1)
        return self.normalize(t)


def build_transform(augment: bool = False) -> transforms.Compose:
    steps: list = [FitPadInvert()]
    if augment:
        steps += [
            transforms.RandomAffine(
                degrees=5, translate=(0.05, 0.05), shear=5, fill=0,
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps.append(ToThreeChannel224())
    # if augment:
    #     steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.08),
    #                                           ratio=(0.3, 3.3), value=0.0))
    return transforms.Compose(steps)


class CrossOutDataset(Dataset):
    """Loads images from {root}/{split}/images/{class}/ for the given class names."""

    def __init__(self, split: str, class_names: Sequence[str],
                 root: Path = DATASET_ROOT, augment: bool = False):
        assert split in {"train", "val", "test"}
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.transform = build_transform(augment=augment)

        self.samples: list[tuple[Path, int]] = []
        split_root = root / split / "images"
        for cname in self.class_names:
            cdir = split_root / cname
            if not cdir.is_dir():
                raise FileNotFoundError(f"Missing class dir: {cdir}")
            idx = self.class_to_idx[cname]
            for p in cdir.iterdir():
                if p.suffix.lower() == ".png":
                    self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        with Image.open(path) as img:
            x = self.transform(img)
        return x, label
