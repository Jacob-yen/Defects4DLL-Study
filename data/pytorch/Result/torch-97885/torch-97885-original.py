import torch
from torch.testing._internal.common_utils import TestCase
import itertools


class TestOptim(TestCase):
    def test_functional_fused_optimizer_with_foundinf(self):
        from torch.optim import adam, adamw

        num_tensors = 5
        for functional_optim, amsgrad, no_grad_scale in itertools.product((adam.adam, adamw.adamw), (False, True),
                                                                          (False, True)):
            params, grads, exp_avgs, exp_avg_sqs = [
                [torch.ones((1,), device="cuda") for _ in range(num_tensors)] for _ in range(4)]
            prev_params = [t.clone().detach() for t in params]
            max_exp_avg_sqs = [torch.ones((1,), device="cuda") for _ in range(num_tensors)] if amsgrad else []
            state_steps = [torch.ones((1,), dtype=torch.float32, device="cuda") for _ in range(num_tensors)]
            grad_scale = None if no_grad_scale else torch.ones((1,), dtype=torch.float32, device="cuda")
            found_inf = torch.ones((1,), dtype=torch.float32, device="cuda")

            functional_optim(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                foreach=False,
                capturable=False,
                fused=True,
                amsgrad=amsgrad,
                beta1=0.9,
                beta2=0.99,
                lr=1e-2,
                weight_decay=0.0,
                eps=1e-8,
                maximize=False,
                grad_scale=grad_scale,
                found_inf=found_inf,
            )

            self.assertEqual(
                state_steps,
                [
                    torch.ones((1,), dtype=torch.float32, device="cuda")
                    for _ in range(num_tensors)
                ],
            )


fx = TestOptim()
fx.test_functional_fused_optimizer_with_foundinf()
