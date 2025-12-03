import torch

from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.testing._internal.common_utils import TestCase


class TestAutograd(TestCase):

    def test_checkpointing_without_reentrant_memory_savings(self):
        class MyModel(nn.Module):
            def __init__(self, n, use_checkpoint, use_reentrant):
                super().__init__()
                self.n = n
                self.use_checkpoint = use_checkpoint
                self.use_reentrant = use_reentrant
                self.layers = nn.ModuleList()
                for i in range(self.n):
                    layer = nn.Sequential(
                        nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256)
                    )
                    self.layers.append(layer)
                # pre-allocate the grad so that increased memory usage is mainly
                # due to activations.
                for layer in self.layers:
                    for lin in layer:
                        lin.weight.grad = torch.ones_like(lin.weight)
                        lin.bias.grad = torch.ones_like(lin.bias)

            def forward(self, x):
                for i in range(self.n):
                    if not self.use_checkpoint:
                        x = self.layers[i](x)
                    else:
                        x = checkpoint(self.layers[i], x, use_reentrant=self.use_reentrant)

                return x

        model_no_checkpoint = MyModel(8, use_checkpoint=False, use_reentrant=False).cuda()
        model_reentrant_checkpoint = MyModel(8, use_checkpoint=True, use_reentrant=True).cuda()
        model_no_reentrant_checkpoint = MyModel(8, use_checkpoint=True, use_reentrant=False).cuda()

        x = torch.randn(100, 256, requires_grad=True, device='cuda')

        torch.cuda.reset_peak_memory_stats()
        loss = model_no_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_checkpoint = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        loss = model_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        loss = model_no_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        self.assertTrue(mem_no_reentrant_checkpoint < mem_no_checkpoint)


fx = TestAutograd()
fx.test_checkpointing_without_reentrant_memory_savings()
