import random
from itertools import product
import torch

torch.set_default_dtype(torch.double)
import torch.nn as nn
from torch.testing._internal.common_nn import NNTestCase


class TestNNDeviceType(NNTestCase):

    def _test_dropout(self, cls, device, input, memory_format=torch.contiguous_format):
        p = 0.2
        input = input.to(device).fill_(1 - p)

        module = cls(p)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        module = cls(p, True)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var + 0)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        # check eval mode doesn't change anything
        for inplace in [True, False]:
            module = cls(p, inplace).eval()
            self.assertEqual(input, module(input))

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def _test_dropout_discontiguous(self, cls, device, memory_format=torch.contiguous_format):
        # In this test, we verify that dropout preserves the layout and data for different memory formats.
        # We check whether, we get same values for the output of dropout, when the probability
        # of dropout is 0 or very close to 0.
        # Reference: https://github.com/pytorch/pytorch/issues/47176
        close_to_zero_p = 1e-10  # Should be almost zero but not zero, as for p=0 different path is taken
        for p in [0, close_to_zero_p]:
            inp = torch.ones(2, 3, 3, 3, device=device)
            inp_discontiguous = torch.empty(2, 3, 3, 6, device=device, memory_format=memory_format)[..., ::2]
            inp_discontiguous.copy_(inp)
            mod = cls(p=p)
            out = mod(inp_discontiguous)
            if p != 0:  # Zero will keep strides as is based on input.
                # When prob == 0, input stride (54, 18, 6, 2) -> output stride (54, 18, 6, 2)
                # When prob != 0, input stride (54, 18, 6, 2) -> output stride (27, 9, 3, 1)
                self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertEqual(inp_discontiguous, out)

    def _test_dropoutNd_channel_zero(self, dropout, input):
        # Verify the number of zeros in a channel is 0 or the number of elements in the channel
        # for a fully positive input tensor
        shape = input.shape
        B = shape[0]
        C = shape[1]
        channel_numel = torch.tensor(shape[2:]).prod()
        result = dropout(input)

        for b, c in product(range(B), range(C)):
            self.assertTrue(result[b, c].count_nonzero() in (0, channel_numel))

    def test_Dropout2d(self, device="cuda"):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        num_features = 1000
        input = torch.empty(num_features, b, w, h)
        self._test_dropout(nn.Dropout2d, device, input)
        self._test_dropout(nn.Dropout2d, device, input, memory_format=torch.channels_last)

        self._test_dropout_discontiguous(nn.Dropout2d, device)
        self._test_dropout_discontiguous(nn.Dropout2d, device, memory_format=torch.channels_last)

        with self.assertWarnsRegex(UserWarning, "Received a 5-D input to dropout2d"):
            nn.Dropout2d(p=0.5)(torch.rand(1, 2, 2, 2, 2, device=device))

        with self.assertWarnsRegex(UserWarning, "Received a 2-D input to dropout2d"):
            nn.Dropout2d(p=0.5)(torch.rand(1, 2, device=device))

        # TODO: Uncomment these lines once no-batch-dim inputs are supported.
        # For now, the historical dropout1d behavior is performed for 3D inputs.
        # See https://github.com/pytorch/pytorch/issues/77081

        # input = torch.rand(50, 2, 2, device=device)
        # self._test_dropoutNd_no_batch(nn.Dropout2d(p=0.5), input)
        # self._test_dropoutNd_no_batch(nn.Dropout2d(p=0.5, inplace=True), input)

        with self.assertWarnsRegex(UserWarning, "assuming that channel-wise 1D dropout behavior is desired"):
            nn.Dropout2d(p=0.5)(torch.rand(1, 2, 2, device=device))


fx = TestNNDeviceType()
fx.test_Dropout2d()
