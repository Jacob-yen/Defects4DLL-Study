import torch
from torch.testing._internal.jit_utils import JitTestCase


class TestScript1(JitTestCase):
    def test_lazy_script(self):
        def foo(x: int):
            return x + 1

        @torch.jit._script_if_tracing
        def fee(x: int = 2):
            return foo(1) + x

        torch.jit.script(fee)


fx = TestScript1()
fx.test_lazy_script()
