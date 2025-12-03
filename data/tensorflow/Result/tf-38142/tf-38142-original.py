# codeStart
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.framework import sparse_tensor
import tensorflow as tf


class SparseTensorTest(test_util.TensorFlowTestCase):

    def testShape(self):
        def test_fn(tensor):
            tensor = sparse_ops.sparse_transpose(tensor)
            self.assertEqual(tensor.shape.rank, 2)
            return tensor


        tensor = sparse_tensor.SparseTensor(
            indices=[[0, 0], [1, 2]], values=[1., 2], dense_shape=[3, 4])
        tf.function(test_fn)(tensor)


fx = SparseTensorTest()
fx.testShape()
# codeEnd
