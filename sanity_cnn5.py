from cnn5 import Cnn5
from dataset import CrossOutDataset
import torch

ds = CrossOutDataset("val", ["CLEAN", "MIXED"], mode="cnn5")
print("samples:", len(ds))
x, y = ds[0]
print("tensor shape:", tuple(x.shape), "dtype:", x.dtype,
      "min/max:", round(float(x.min()), 3), round(float(x.max()), 3))

m = Cnn5(num_classes=2)
m.eval()
with torch.no_grad():
    out = m(x.unsqueeze(0))
print("logits shape:", tuple(out.shape))

n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"trainable params: {n_params:,}")
print("DONE")
