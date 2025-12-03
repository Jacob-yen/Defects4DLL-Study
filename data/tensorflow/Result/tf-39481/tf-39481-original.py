# codeStart
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops


class DivAndModTest(test_util.TensorFlowTestCase):

    def testWithPythonValue(self):
        x = math_ops.divide(5, 2)
        self.assertTrue(isinstance(x, ops.Tensor))


fx = DivAndModTest()
fx.testWithPythonValue()
# codeEnd
