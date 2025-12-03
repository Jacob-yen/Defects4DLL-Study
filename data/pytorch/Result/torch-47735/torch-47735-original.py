import torch
from unittest import TestCase
import numpy as np
import pickle
class SubTensor2(torch.Tensor):
    pass


class TestNamedTuple(TestCase):

    def test_max(self):
        x = torch.tensor([1, 2], device='cpu', dtype=torch.float32)
        xs = x.as_subclass(SubTensor2)
        r = torch.max(x, dim=0)
        rs = torch.max(xs, dim=0)
        self.assertEqual(type(r), type(rs))


testClass = TestNamedTuple()
testClass.test_max()
