import unittest
import torch
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_arange_with_floats_out(self):
        class ArangeModelEnd(torch.nn.Module):
            def forward(self, end):
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(end, out=out_t)

        y = torch.tensor(8.5, dtype=torch.float)
        run_test(ArangeModelEnd(), (y))


fx = TestONNXRuntime()
setUp()
fx.test_arange_with_floats_out()
