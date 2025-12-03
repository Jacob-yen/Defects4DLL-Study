import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase


class TestModuleHooks(TestCase):
    def test_full_backward_pre_hooks(self):
        # Backward pre hook can affect subsequent gradient computation
        a = torch.ones(2, requires_grad=True)
        model = nn.Linear(2, 2)

        def fn(_unused_module, grad_output):
            return (grad_output[0] * 0,)

        model.register_full_backward_pre_hook(fn)

        out = model(a)
        out.sum().backward()
        self.assertEqual(a.grad, torch.zeros_like(a))


fx = TestModuleHooks()
fx.test_full_backward_pre_hooks()
