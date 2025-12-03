import torch
import torch.nn as nn
from torch.distributed._composable import contract
from torch.testing._internal.common_utils import TestCase


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.p = nn.Parameter(torch.randn(10, 10), requires_grad=True)
        self.b = torch.zeros(1)  # buffer

    def forward(self, x, y):
        with torch.no_grad():
            self.b += x.sum() + y.sum()

        return self.p + self.seq1(x) + self.seq2(y)


class TestContract(TestCase):
    def test_modify_fqn(self):
        class ModelWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x):
                return self.module(x)

        @contract()
        def wrap_module(module: nn.Module) -> nn.Module:
            return ModelWrapper(module)

        model = ToyModel()

        with self.assertRaisesRegex(
                RuntimeError,
                "Check parameters, Composable distributed API implementations cannot modify FQNs",
        ):
            wrap_module(model.seq1)


fx = TestContract()
fx.test_modify_fqn()
