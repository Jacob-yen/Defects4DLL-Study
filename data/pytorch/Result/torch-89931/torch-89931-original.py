import torch
import torch.ao.quantization
from torch.testing._internal.common_utils import TestCase

class TestQConfig(TestCase):

    REDUCE_RANGE_DICT = {
        'qnnpack': (False, False)
    }

    def test_reduce_range_qat(self) -> None:
        for backend, reduce_ranges in self.REDUCE_RANGE_DICT.items():
            for version in range(2):
                qconfig = torch.ao.quantization.get_default_qat_qconfig(backend, version)

                fake_quantize_activ = qconfig.activation()
                self.assertEqual(fake_quantize_activ.activation_post_process.reduce_range, reduce_ranges[0])


fx = TestQConfig()
fx.test_reduce_range_qat()
