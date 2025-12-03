import unittest
import torch
import numpy as np
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_chunk(self):
        class ChunkModel(torch.nn.Module):
            def __init__(self, dim=1):
                super(ChunkModel, self).__init__()
                self.dim = dim

            def forward(self, x):
                return torch.chunk(x, 3, dim=self.dim)

        model = ChunkModel()
        model.eval()
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        x = torch.randn(1, 18)

        for dim_size_ in range(13, 16):
            y = torch.randn(1, dim_size_)

            run_test(model_neg_dim, x, test_with_inputs=[y],
                     input_names=['x'],
                     dynamic_axes={'x': {0: 'batch_size', 1: 'dims'}})


fx = TestONNXRuntime()
setUp()
fx.test_chunk()
