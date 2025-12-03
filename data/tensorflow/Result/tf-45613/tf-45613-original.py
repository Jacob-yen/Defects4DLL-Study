import tensorflow
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.python.ops import image_ops


class CentralCropTest(test_util.TensorFlowTestCase):
    def testCentralFractionTensor(self):
        # Test case for GitHub issue 45324.
        x_shape = [240, 320, 3]

        @def_function.function(autograph=False)
        def f(x, central_fraction):
            return image_ops.central_crop(x, central_fraction)

        x_np = np.zeros(x_shape, dtype=np.int32)
        c_np = constant_op.constant(0.33)
        self.evaluate(f(x_np, c_np))


testClass = CentralCropTest()
testClass.testCentralFractionTensor()
# codeEnd
