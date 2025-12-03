from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
import tensorflow

class RangeTest(test.TestCase):
    def testMixedDType(self):
        with self.cached_session(use_gpu=True):
            constant = constant_op.constant(5)
            math_ops.range(constant, dtype=dtypes.float32)

fx = RangeTest()
fx.testMixedDType()
