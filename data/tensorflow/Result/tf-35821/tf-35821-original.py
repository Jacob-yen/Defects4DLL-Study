import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
import tensorflow

class RangeTest(test.TestCase):
    def testMixedDType(self):
        constant = constant_op.constant(4, dtype=dtypes.int32)
        math_ops.range(constant, dtype=dtypes.int64)


fx = RangeTest()
fx.testMixedDType()
