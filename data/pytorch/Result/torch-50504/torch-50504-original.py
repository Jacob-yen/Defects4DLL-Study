import unittest
import torch
import numpy as np
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_unfold_dynamic_inputs(self):
        setUp()

        class UnfoldModel(torch.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=x.shape[1], step=1)

        x = torch.randn(4, 2, 4, requires_grad=True)
        run_test(UnfoldModel(), x)


fx = TestONNXRuntime()
fx.test_unfold_dynamic_inputs()
