import tensorflow
from absl.testing import parameterized
from tensorflow.python import keras
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.platform import test


class GlobalPoolingTest(test.TestCase, parameterized.TestCase):

  @testing_utils.enable_v2_dtype_behavior
  def test_mixed_float16_policy(self):
    with policy.policy_scope('mixed_float16'):
      inputs1 = keras.Input(shape=(36, 512), dtype="float16")
      inputs2 = keras.Input(shape=(36,), dtype="bool")
      average_layer = keras.layers.pooling.GlobalAveragePooling1D()
      average_layer(inputs1, inputs2)


fx = GlobalPoolingTest()
fx.test_mixed_float16_policy()
# codeEnd
