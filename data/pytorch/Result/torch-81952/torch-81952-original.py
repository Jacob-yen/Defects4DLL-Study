import torch

model = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
src = torch.rand(32, 10, 512)
src_mask = torch.zeros(10, 10).to(torch.bool)

model.eval()
with torch.no_grad():
    print(model(src, src_mask))