import unittest
import torch
import numpy as np
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_empty_constant_shape(self):
        class Zeros(torch.nn.Module):
            def forward(self, x):
                y = torch.zeros(())
                y += x
                return y

        x = torch.tensor(42.)
        run_test(Zeros(), x)


fx = TestONNXRuntime()
setUp()
fx.test_empty_constant_shape()
