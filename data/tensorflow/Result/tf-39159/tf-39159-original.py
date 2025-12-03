from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
import tensorflow
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
import numpy as np

class BooleanMaskTest(test_util.TensorFlowTestCase):

    def testMaskWithAxisTensor(self):
        @def_function.function(autograph=False)
        def f():
            tensor = [1, 2, 3]
            mask = [True, False, True]
            constant = constant_op.constant(0, dtype=dtypes.int32)
            return array_ops.boolean_mask(tensor, mask, axis=constant)

        self.evaluate(f())


fx = BooleanMaskTest()
fx.testMaskWithAxisTensor()
