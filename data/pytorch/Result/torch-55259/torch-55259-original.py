import unittest
import torch
from typing import List
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_prim_min(self):
        @torch.jit.script
        def list_append(boxes: List[torch.Tensor]):
            temp = []
            for i, b in enumerate(boxes):
                temp.append(torch.full_like(b[:, 1], i))
            return temp[0]

        class Min(torch.nn.Module):
            def forward(self, x):
                boxes = [x, x, x]
                return list_append(boxes)

        x = torch.rand(5, 5)
        run_test(Min(), (x,))


fx = TestONNXRuntime()
setUp()
fx.test_prim_min()
