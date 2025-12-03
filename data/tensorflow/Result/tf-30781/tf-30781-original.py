from absl.testing import parameterized
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops


class SparseOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
    def testSparseTensorToDenseString(self):
        sp = sparse_tensor.SparseTensor(
            indices=[[0, 0], [1, 2]],
            values=['a', 'b'],
            dense_shape=[2, 3])
        sparse_ops.sparse_tensor_to_dense(sp)


partitionedCallTest = SparseOpsTest()
partitionedCallTest.testSparseTensorToDenseString()
