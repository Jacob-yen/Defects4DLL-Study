import unittest
import torch
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_dynamic_repeat_interleave(self):
        class SingleDynamicModel(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor(4)
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        run_test(SingleDynamicModel(), x, test_with_inputs=[another_x], input_names=['input_1'],
                 dynamic_axes={'input_1': {1: 'w'}})


fx = TestONNXRuntime()
setUp()
fx.test_dynamic_repeat_interleave()
