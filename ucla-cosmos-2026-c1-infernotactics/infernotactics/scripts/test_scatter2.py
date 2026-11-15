import torch
import torch_directml

device = torch_directml.device()
print('device:', device)
x = torch.zeros(2, 3, device=device)
print('shape:', x.shape, file=open('test_out.txt', 'w'))
print('ndim:', x.ndim, file=open('test_out.txt', 'a'))
print('device:', x.device, file=open('test_out.txt', 'a'))