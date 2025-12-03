import torch
from torch.quantization._numeric_suite import OutputLogger

from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.testing._internal.common_quantized import override_qengines


class TestEagerModeNumericSuite(QuantizationTestCase):
    @override_qengines
    def test_output_logger(self):
        r"""Compare output from OutputLogger with the expected results"""
        x = torch.rand(2, 2)
        y = torch.rand(2, 1)

        l = []
        l.append(x)
        l.append(y)

        logger = OutputLogger()
        logger.forward(x)
        logger.forward(y)


fx = TestEagerModeNumericSuite()
fx.test_output_logger()
