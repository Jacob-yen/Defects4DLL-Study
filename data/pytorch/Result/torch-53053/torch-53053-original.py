import unittest
import torch
from util import run_test, setUp


class TestONNXRuntime(unittest.TestCase):

    def test_embedding(self):
        class EmbedModel(torch.nn.Module):
            def forward(self, input, emb):
                return torch.nn.functional.embedding(input, emb, padding_idx=1)

        model = EmbedModel()
        x = torch.randint(4, (4,))
        x[2] = x[0] = 1
        embedding_matrix = torch.rand(10, 3)
        run_test(model, (x, embedding_matrix))


fx = TestONNXRuntime()
setUp()
fx.test_embedding()
