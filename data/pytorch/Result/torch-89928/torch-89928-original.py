from torch.testing._internal.common_utils import TestCase
import torch


class TestAutograd(TestCase):

    def test_full_backward_hook_double_backward(self):
        x = torch.rand(1, requires_grad=True)
        y = torch.rand_like(x)

        func = torch.nn.MSELoss()
        counter = [0]

        def hook(module, grad_input, grad_output):
            counter[0] += 1

        func.register_full_backward_hook(hook)

        f = func(x, y)

        (gradx_f,) = torch.autograd.grad(f, x, create_graph=True)
        torch.autograd.grad(gradx_f, x)


fx = TestAutograd()
fx.test_full_backward_hook_double_backward()
