import torch


ckpt = torch.load('checkpoints/autovc.ckpt', map_location=torch.device('cpu'))
