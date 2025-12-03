import unittest
from util import run_test, setUp
import torch
import numpy as np
import io
import copy
import os


class TestONNXRuntime(unittest.TestCase):

    def test_scatter_add(self):
        @torch.jit.script
        def scatter_sum(src: torch.Tensor, index: torch.Tensor):
            size = src.size()
            out = torch.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(torch.nn.Module):
            def forward(self, src, index):
                return scatter_sum(src, index)

        src = torch.rand(3, 2)
        index = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int64)
        run_test(ScatterModel(), (src, index))


fx = TestONNXRuntime()
setUp()
fx.test_scatter_add()
