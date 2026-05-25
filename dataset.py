import hashlib
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

    def __init__(self, target_w: int = 136, target_h: int = 68):
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


class ToSingleChannel:
    """Grayscale PIL at native FitPadInvert size → 1xHxW tensor in [0,1]."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return TF.to_tensor(img)  # 1xHxW


class ThumbnailPadWhite:
    """RGB PIL -> RGB PIL of size (size, size), downscaled to fit (no upscaling)
    and centered on a white canvas. Matches the YOLOClassifier training pipeline
    (PIL.Image.thumbnail + paste on white).
    """

    def __init__(self, size: int = 128):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        img.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        w, h = img.size
        canvas.paste(img, ((self.size - w) // 2, (self.size - h) // 2))
        return canvas


class InvertTensor:
    """t -> 1 - t."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t


YOLO_NORM_MEAN = [0.0608, 0.0608, 0.0608]
YOLO_NORM_STD = [0.1927, 0.1927, 0.1927]


def build_transform(augment: bool = False, mode: str = "convnext") -> transforms.Compose:
    """mode='convnext' → 3-channel 224x224 ImageNet-normalized.
    mode='cnn5'      → 1-channel 136x68 in [0,1].
    mode='yolo128'   → 3-channel 128x128, thumbnail+pad-white, inverted,
                       normalized with YOLOClassifier training stats."""
    assert mode in {"convnext", "cnn5", "yolo128"}
    if mode == "yolo128":
        steps: list = [
            ThumbnailPadWhite(size=128),
            transforms.ToTensor(),
            InvertTensor(),
            transforms.Normalize(YOLO_NORM_MEAN, YOLO_NORM_STD),
        ]
        return transforms.Compose(steps)

    steps = [FitPadInvert()]
    if augment:
        steps += [
            transforms.RandomAffine(
                degrees=5, translate=(0.05, 0.05), shear=5, fill=0,
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    if mode == "convnext":
        steps.append(ToThreeChannel224())
    else:
        steps.append(ToSingleChannel())
    return transforms.Compose(steps)

# def build_transform(augment: bool = False, mode: str = "convnext") -> transforms.Compose:
#     """mode='convnext' → 3-channel 224x224 ImageNet-normalized.
#     mode='cnn5'      → 1-channel 136x68 in [0,1]."""
#     assert mode in {"convnext", "cnn5"}
#     steps: list = []
#     if augment:
#         steps += [
#             transforms.RandomAffine(
#                 degrees=5, translate=(0.1, 0.1), shear=5, fill=1,
#             ),
#             transforms.ColorJitter(brightness=0.3, contrast=0.3),
#         ]
#     steps.append(FitPadInvert())
#     if mode == "convnext":
#         steps.append(ToThreeChannel224())
#     else:
#         steps.append(ToSingleChannel())
#     return transforms.Compose(steps)

CLEAN_CLASS = "CLEAN"


def _pixel_hash(path: Path) -> str:
    with Image.open(path) as im:
        buf = im.convert("L").tobytes()
    return hashlib.md5(buf).hexdigest()


def _load_or_build_duplicate_set(split_root: Path) -> set[str]:
    """Return filenames of non-CLEAN images whose pixels equal their CLEAN twin.

    Result is cached under {split_root}/.duplicate_clean.txt; the cache is rebuilt
    if missing. Delete that file to force a rescan.
    """
    cache = split_root / ".duplicate_clean.txt"
    if cache.exists():
        return {line.strip() for line in cache.read_text().splitlines() if line.strip()}

    clean_dir = split_root / CLEAN_CLASS
    if not clean_dir.is_dir():
        return set()

    clean_hashes = {p.name: _pixel_hash(p) for p in clean_dir.iterdir() if p.suffix.lower() == ".png"}

    bad: set[str] = set()
    for cdir in split_root.iterdir():
        if not cdir.is_dir() or cdir.name == CLEAN_CLASS:
            continue
        for p in cdir.iterdir():
            if p.suffix.lower() != ".png":
                continue
            ch = clean_hashes.get(p.name)
            if ch is not None and _pixel_hash(p) == ch:
                bad.add(f"{cdir.name}/{p.name}")

    cache.write_text("\n".join(sorted(bad)))
    return bad


class CrossOutDataset(Dataset):
    """Loads images from {root}/{split}/images/{class}/ for the given class names.

    Drops non-CLEAN samples whose pixels are identical to the same-named CLEAN
    image — these are mislabeled because the synthetic cross-out generator failed
    on small inputs and emitted the unmodified clean image.
    """

    def __init__(self, split: str, class_names: Sequence[str],
                 root: Path = DATASET_ROOT, augment: bool = False,
                 mode: str = "convnext"):
        assert split in {"train", "val", "test"}
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.transform = build_transform(augment=augment, mode=mode)

        split_root = root / split / "images"
        bad = _load_or_build_duplicate_set(split_root)

        self.samples: list[tuple[Path, int]] = []
        for cname in self.class_names:
            cdir = split_root / cname
            if not cdir.is_dir():
                raise FileNotFoundError(f"Missing class dir: {cdir}")
            idx = self.class_to_idx[cname]
            is_clean = cname == CLEAN_CLASS
            for p in cdir.iterdir():
                if p.suffix.lower() != ".png":
                    continue
                if not is_clean and f"{cname}/{p.name}" in bad:
                    continue
                self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        with Image.open(path) as img:
            x = self.transform(img)
        return x, label
