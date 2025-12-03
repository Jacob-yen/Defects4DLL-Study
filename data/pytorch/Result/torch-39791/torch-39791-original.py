import unittest
import torch
import numpy as np
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_pow(self):
        class PowModule(torch.nn.Module):
            def forward(self, x, y):
                return x.pow(y)

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 3, 4)).to(dtype=torch.int32)
        run_test(PowModule(), (x, y))


fx = TestONNXRuntime()
setUp()
fx.test_pow()
