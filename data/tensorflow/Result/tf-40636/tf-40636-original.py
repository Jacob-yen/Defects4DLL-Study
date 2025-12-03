import tensorflow
from tensorflow.python.framework import constant_op
from absl.testing import parameterized
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops


class SparseOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    def testConstantStringToSparse(self):
        # Test case for GitHub issue 40633.
        tensor = constant_op.constant(list("ababa"))
        sparse_ops.from_dense(tensor)


fx = SparseOpsTest()
fx.testConstantStringToSparse()
# codeEnd
