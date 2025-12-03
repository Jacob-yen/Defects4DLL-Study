import torch
import numpy as np
import io
import copy
import os
import unittest
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_remainder_scalar(self):
        class RemainderModel(torch.nn.Module):

            def __init__(self, scalar=2.55):
                super().__init__()
                self.scalar = scalar

            def forward(self, input):
                return torch.remainder(input, self.scalar)

        x = torch.tensor([7, 6, -7, -6], dtype=torch.long)
        run_test(RemainderModel(2), x)


fx = TestONNXRuntime()
setUp()
fx.test_remainder_scalar()
