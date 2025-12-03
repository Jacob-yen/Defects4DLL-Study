import unittest
import torch
import numpy as np
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_index_put_singular(self):
        class IndexPutBoolModel(torch.nn.Module):
            def forward(self, mask, indices):
                mask[indices] = True
                return mask

        mask = torch.zeros(100, dtype=torch.bool)
        indices = (torch.rand(25) * mask.shape[0]).to(torch.int64)
        run_test(IndexPutBoolModel(), (mask, indices))


fx = TestONNXRuntime()
setUp()
fx.test_index_put_singular()
