# codeStart
from tensorflow.python.ops import histogram_ops
from tensorflow.python.platform import test
import numpy as np


class BinValuesFixedWidth(test.TestCase):

    def test_range_overlap(self):
        # GitHub issue 29661
        value_range = np.float32([0.0, 0.0])
        values = np.float32([-1.0, 0.0, 1.5, 2.0, 5.0, 15])
        expected_bins = [0, 0, 4, 4, 4, 4]
        with self.assertRaises(ValueError):
            with self.cached_session():
                _ = histogram_ops.histogram_fixed_width_bins(
                    values, value_range, nbins=5)


fx = BinValuesFixedWidth()
fx.test_range_overlap()
# codeEnd

