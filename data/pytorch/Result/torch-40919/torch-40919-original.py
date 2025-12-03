import torch
from torch.testing._internal.common_utils import TestCase


class _TestTorchMixin(TestCase):

    def test_print(self):
        x = torch.tensor([2.3 + 4.0j, 7 + 6.0j])
        x_str = str(x)
        self.assertExpectedInline(x_str, 'tensor([2.3000+4.j, 7.0000+6.j])')


fx = _TestTorchMixin()
fx.test_print()
