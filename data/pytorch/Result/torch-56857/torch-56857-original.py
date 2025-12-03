import unittest
import torch
import numpy as np
import os
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):


    def test_to_device(self):


        class M_ToDevice(torch.nn.Module):

            def forward(self, x, y):
                return x.to(y.device), y
        x = torch.randn(6)
        y = torch.randn(6)
        run_test(M_ToDevice(), (x, y))


fx = TestONNXRuntime()
setUp()
fx.test_to_device()
