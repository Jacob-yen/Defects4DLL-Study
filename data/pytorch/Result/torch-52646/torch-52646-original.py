import unittest
import onnxruntime
import torch
import numpy as np
import os
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_pow(self):
        class PowModule2(torch.nn.Module):

            def forward(self, x):
                return torch.pow(2, x)

        x = torch.randn(1, 10)
        run_test(PowModule2(), (x,))


fx = TestONNXRuntime()
setUp()
fx.test_pow()
