import unittest
import torch
from util import run_test, setUp
import numpy as np
import os


class TestONNXRuntime(unittest.TestCase):

    def test_index_put_slice_index(self):
        class IndexPutModel9(torch.nn.Module):

            def forward(self, poses):
                w = 32
                x = poses[:, :, 0] - (w - 1) // 2
                boxes = torch.zeros([poses.shape[0], 17, 4])
                boxes[:, :, 0] = x
                return boxes

        x = torch.zeros([2, 17, 3], dtype=torch.int64)
        run_test(IndexPutModel9(), (x,))


fx = TestONNXRuntime()
setUp()
fx.test_index_put_slice_index()
