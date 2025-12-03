import torch

data = {"a": 1, "b": "str", "c": 1e-6}
torch.save(data, "data.pth")
torch.load("data.pth", weights_only=True)
