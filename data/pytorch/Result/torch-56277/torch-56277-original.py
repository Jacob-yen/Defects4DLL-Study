import unittest
import torch
import numpy as np
from util import run_test, setUp
import os


class TestONNXRuntime(unittest.TestCase):

    def test_split_dynamic_axes(self):
        class Split(torch.nn.Module):

            def forward(self, x):
                return x.split(1, dim=-1)

        x = torch.randn(4, 384, 2)
        input_names = ['logits']
        run_test(Split(), x, input_names=input_names, dynamic_axes={input_names[0]: {(0): 'batch'}})


fx = TestONNXRuntime()
setUp()
fx.test_split_dynamic_axes()
