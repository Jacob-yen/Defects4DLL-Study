import torch
from torch import fx
from typing import Any, Callable, Dict
from torch.testing._internal.jit_utils import JitTestCase
import unittest


class TestFX(JitTestCase):

    def test_trace_dict_int_keys(self):


        class ModWithDictArg(torch.nn.Module):

            def forward(self, d: Dict[int, torch.Tensor]):
                return d[42]


        class CallsModWithDict(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.m = ModWithDictArg()

            def forward(self, x):
                return self.m({(42): x})


        class MyTracer(fx.Tracer):

            def is_leaf_module(self, m: torch.nn.Module,
                module_qualified_name: str) ->bool:
                return isinstance(m, ModWithDictArg)
        traced_graph = MyTracer().trace(CallsModWithDict())


tfx = TestFX()
tfx.test_trace_dict_int_keys()
