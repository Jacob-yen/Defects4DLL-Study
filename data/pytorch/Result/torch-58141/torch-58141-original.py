import unittest
import torch
import numpy as np
import os
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_sum_empty_tensor(self):
        class M(torch.nn.Module):

            def forward(self, x):
                return x[0:0].sum()

        x = torch.ones(12)
        run_test(M(), (x,))


fx = TestONNXRuntime()
setUp()
fx.test_sum_empty_tensor()