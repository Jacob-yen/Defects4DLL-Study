import unittest
import torch
import numpy as np
from util import run_model_test_with_external_data, setUp


class TestONNXRuntime(unittest.TestCase):
    def test_largemodel_without_use_external_data_format_param(self):
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super(LargeModel, self).__init__()
                dim = 5
                n = 40 * 4 * 10 ** 6
                self.emb = torch.nn.Embedding(n, dim)
                self.lin1 = torch.nn.Linear(dim, 1)
                self.seq = torch.nn.Sequential(
                    self.emb,
                    self.lin1,
                )

            def forward(self, input):
                return self.seq(input)

        model = LargeModel()
        x = torch.tensor([2], dtype=torch.long)
        run_model_test_with_external_data(LargeModel(), x, use_external_data_format=None)


fx = TestONNXRuntime()
setUp()
fx.test_largemodel_without_use_external_data_format_param()
