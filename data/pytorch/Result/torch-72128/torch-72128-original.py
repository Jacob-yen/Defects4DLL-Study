import unittest
import torch
import numpy as np
import io
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_cosine_similarity(self):
        x = torch.randn(5, 3, 2)
        y = torch.randn(5, 3, 2)
        run_test(torch.nn.CosineSimilarity(dim=2), input=(x, y))


pc = TestONNXRuntime()
setUp()
pc.test_cosine_similarity()
