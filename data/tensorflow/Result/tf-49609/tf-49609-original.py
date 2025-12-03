import tensorflow
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops


@test_util.run_all_in_graph_and_eager_modes
class ReduceTest(test_util.TensorFlowTestCase):
    def testReduceVar(self):
        x = ragged_factory_ops.constant([[5., 1., 4., 1.], [], [5., 9., 2.], [5.], []])
        math_ops.reduce_variance(x, axis=0)


testClass = ReduceTest()
testClass.testReduceVar()
# codeEnd
