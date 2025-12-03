from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class NumpyFunctionTest(test.TestCase):
    @test_util.run_in_graph_and_eager_modes
    def test_numpy_arguments(self):
        def plus(a, b):
            return a + b

        script_ops.numpy_function(plus, [1, 2], dtypes.int32)


partitionedCallTest = NumpyFunctionTest()
partitionedCallTest.test_numpy_arguments()
