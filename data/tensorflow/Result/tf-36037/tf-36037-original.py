from absl.testing import parameterized
from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
import tensorflow

@test_util.run_all_in_graph_and_eager_modes
class BackendLinearAlgebraTest(test.TestCase, parameterized.TestCase):
    def test_relu(self):
        x = keras.Input(shape=(), name='x', dtype='int64')
        keras.layers.ReLU(max_value=100, dtype='int64')(x)


fx = BackendLinearAlgebraTest()
fx.test_relu()
# codeEnd
