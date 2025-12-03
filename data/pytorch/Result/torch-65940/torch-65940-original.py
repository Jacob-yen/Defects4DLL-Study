import unittest
import torch
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_einsum(self):
        class EinsumModelBatchDiagonal(torch.nn.Module):
            def forward(self, x):
                eqn = "...ii ->...i"
                return torch.einsum(eqn, x)

        for x in [torch.randn(3, 5, 5), torch.randn(3, 5, 5).to(dtype=torch.bool)]:
            run_test(EinsumModelBatchDiagonal(), input=(x,))


fx = TestONNXRuntime()
setUp()
fx.test_einsum()
