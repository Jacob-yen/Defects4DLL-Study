import unittest
import torch
import numpy as np
from typing import List
from util import run_test, setUp
import copy

class TestONNXRuntime(unittest.TestCase):
    def test_pad_types(self):
        class Pad(torch.nn.Module):

            def forward(self, x, pad: List[int]):
                return torch.nn.functional.pad(x, pad)

        x = torch.randn(2, 2, 4, 4)
        y = pad = [2, 4]
        run_test(Pad(), (x, y))


a = TestONNXRuntime()
setUp()
a.test_pad_types()
