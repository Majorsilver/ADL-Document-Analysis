from dataset import CrossOutDataset
from train import build_model
import torch
print('imports ok', flush=True)
ds = CrossOutDataset('val', ['CLEAN', 'MIXED'])
print('val binary samples:', len(ds), flush=True)
x, y = ds[0]
print('tensor shape:', tuple(x.shape), 'label:', y,
      'range:', round(float(x.min()), 3), round(float(x.max()), 3), flush=True)
m = build_model(2)
m.eval()
with torch.no_grad():
    out = m(x.unsqueeze(0))
print('logits shape:', tuple(out.shape), flush=True)
print('DONE', flush=True)
