import torch
import torch.optim as optim
import torch.optim._multi_tensor as optim_mt
from torch.testing._internal.common_utils import TestCase


class TestOptim(TestCase):
    exact_dtype = True

    def _test_complex_optimizer(self, optimizer_constructor):
        complex_param = torch.randn(5, 5, dtype=torch.complex64, requires_grad=True)
        real_param = torch.view_as_real(complex_param).detach().clone().requires_grad_()
        complex_opt = optimizer_constructor(complex_param)
        real_opt = optimizer_constructor(real_param)

        for i in range(3):
            complex_param.grad = torch.randn_like(complex_param)
            real_param.grad = torch.view_as_real(complex_param.grad)
            complex_opt.step()
            real_opt.step()
            self.assertEqual(torch.view_as_real(complex_param), real_param)

    def test_adagrad_complex(self):
        for optimizer in [optim.Adagrad, optim_mt.Adagrad]:
            self._test_complex_optimizer(
                lambda param: optimizer([param], lr=1e-1)
            )
            self._test_complex_optimizer(
                lambda param: optimizer(
                    [param], lr=1e-1, initial_accumulator_value=0.1
                )
            )


fx = TestOptim()
fx.test_adagrad_complex()
