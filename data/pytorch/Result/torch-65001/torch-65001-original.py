import unittest
import torch
import numpy as np
import os
from util import run_test, setUp
import pickle


class TestONNXRuntime(unittest.TestCase):

    def test_softplus(self):
        class BetaModel(torch.nn.Module):

            def forward(self, x):
                return torch.nn.functional.softplus(x, beta=2)

        x = torch.randn(3, 4, 5, requires_grad=True)
        run_test(BetaModel(), x)


fx = TestONNXRuntime()
setUp()
fx.test_softplus()
