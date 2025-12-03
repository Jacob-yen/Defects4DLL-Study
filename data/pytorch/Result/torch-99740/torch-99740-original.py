import torch

import torch._dynamo
import torch._dynamo.test_case


class AotAutogradFallbackTests(torch._dynamo.test_case.TestCase):
    def test_mutation(self):
        def fn(param, y):
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(False)
                param.add_(y)
            finally:
                torch.set_grad_enabled(prev_grad)
            return y

        y = torch.randn(4)
        x = torch.nn.Parameter(torch.randn(4))
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        # This should not error: we mutated an autograd leaf under no_grad mode.
        aot_fn(x, y)


fx = AotAutogradFallbackTests()
fx.test_mutation()
