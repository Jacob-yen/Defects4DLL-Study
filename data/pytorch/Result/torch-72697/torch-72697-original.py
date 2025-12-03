import torch
from torch.testing._internal.common_utils import TestCase


class PackedSequenceTest(TestCase):
    def test_pad_sequence_with_tensor_sequences(self):
        tensor = torch.tensor([[[7, 6]], [[-7, -1]]])
        torch.nn.utils.rnn.pad_sequence(tensor)


fx = PackedSequenceTest()
fx.test_pad_sequence_with_tensor_sequences()
