import sys
sys.path.insert(0, r'A:\AI\InfernoTactics\infernotactics\src')
import torch
import torch_directml

device = torch_directml.device()
print('Testing scatter...', flush=True)
x = torch.zeros(2, 3, device=device)
idx = torch.tensor([0, 1], device=device).unsqueeze(1)
try:
    x.scatter_(1, idx, torch.tensor([1.0, 2.0], device=device))
    print('OK', x, flush=True)
except Exception as e:
    print('FAIL:', e, flush=True)