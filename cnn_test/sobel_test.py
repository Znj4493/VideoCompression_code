import torch
import torch.nn.functional as F

sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
print(sobel_x.shape)