import torch.nn as nn
import torch
import onnxruntime as ort


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 64)

    def forward(self, x):
        inp = x.new_zeros(x.shape)

        return self.emb(inp)


model = MyModel()
inp = torch.Tensor([[2, 5, 6], [3, 2, 5]]).to(torch.int64)

torch.onnx.export(model, (inp,), "model.onnx", opset_version=9)

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])