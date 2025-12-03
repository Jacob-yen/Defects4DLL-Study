import unittest
import torch
import numpy as np
import os
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_reduced_sum_dtypes(self):
        class NoDimModel(torch.nn.Module):

            def forward(self, input):
                return input.sum(dtype=torch.float)

        input = torch.randn((4, 4), dtype=torch.half)
        run_test(NoDimModel(), input)


fx = TestONNXRuntime()
setUp()
fx.test_reduced_sum_dtypes()
