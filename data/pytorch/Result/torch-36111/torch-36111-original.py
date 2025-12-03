import torch
import unittest
from typing import Optional


class TestScript(unittest.TestCase):

    def test_torchbind_optional_explicit_attr(self):


        class TorchBindOptionalExplicitAttr(torch.nn.Module):
            foo: Optional[torch.classes._TorchScriptTesting._StackString]

            def __init__(self):
                super().__init__()
                self.foo = torch.classes._TorchScriptTesting._StackString([
                    'test'])

            def forward(self) ->str:
                foo_obj = self.foo
                if foo_obj is not None:
                    return foo_obj.pop()
                else:
                    return '<None>'
        mod = TorchBindOptionalExplicitAttr()
        torch.jit.script(mod)


fx = TestScript()
fx.test_torchbind_optional_explicit_attr()
