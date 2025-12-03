from itertools import product
import torch
import torch.cuda
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.autocast_test_lists import AutocastTestLists


class TestCuda(TestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    FIFTY_MIL_CYCLES = 50000000

    def setUp(self):
        super().setUp()
        self.autocast_lists = AutocastTestLists(torch.device('cuda:0'))

    def _create_scaling_models_optimizers(self, device="cuda", optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
        # Create a module+optimizer that will use scaling, and a control module+optimizer
        # that will not use scaling, against which the scaling-enabled module+optimizer can be compared.
        mod_control = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
        mod_scaling = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
        with torch.no_grad():
            for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
                s.copy_(c)

        kwargs = {"lr": 1.0}
        if optimizer_kwargs is not None:
            kwargs.update(optimizer_kwargs)
        opt_control = optimizer_ctor(mod_control.parameters(), **kwargs)
        opt_scaling = optimizer_ctor(mod_scaling.parameters(), **kwargs)

        return mod_control, mod_scaling, opt_control, opt_scaling

    def _create_scaling_case(self, device="cuda", dtype=torch.float, optimizer_ctor=torch.optim.SGD,
                             optimizer_kwargs=None):
        data = [(torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
                (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
                (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
                (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device))]

        loss_fn = torch.nn.MSELoss().cuda()

        skip_iter = 2

        return self._create_scaling_models_optimizers(
            device=device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optimizer_kwargs,
        ) + (data, loss_fn, skip_iter)

    def test_params_invalidated_with_grads_invalidated_between_unscale_and_step(self):
        for optimizer_ctor, optimizer_kwargs in product(
                (torch.optim.Adam, torch.optim.AdamW),
                (
                        {"foreach": False, "fused": False},
                        {"foreach": True, "fused": False},
                        {"foreach": False, "fused": True},
                ),
        ):
            with self.subTest(optimizer=optimizer_ctor, optimizer_kwargs=optimizer_kwargs):
                self._test_grads_invalidated_between_unscale_and_step(optimizer_ctor, optimizer_kwargs)

    def _test_grads_invalidated_between_unscale_and_step(self, optimizer_ctor, optimizer_kwargs):
        model, _, optimizer, _, data, loss_fn, _ = self._create_scaling_case(
            optimizer_ctor=optimizer_ctor, optimizer_kwargs=optimizer_kwargs,
        )
        scaler = torch.cuda.amp.GradScaler(init_scale=128.0)

        for input, target in data:
            optimizer.zero_grad()
            with torch.autocast('cuda', enabled=True):
                output = model(input)
                loss = loss_fn(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # deliberately break grads
            for j, param in enumerate(model.parameters()):
                param.grad.copy_(torch.inf if j % 2 else torch.nan)

            scaler.step(optimizer)
            scaler.update()

        self.assertTrue(all((p.isnan().any() or p.isinf().any()) for p in model.parameters()))


fx = TestCuda()
fx.setUp()
fx.test_params_invalidated_with_grads_invalidated_between_unscale_and_step()
