import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers import normalization
from tensorflow.python.keras.layers import normalization_v2


class BatchNormalizationV2Test(keras_parameterized.TestCase):

    def test_basic_batchnorm_v2_none_shape_and_virtual_batch_size(self):
        norm = normalization_v2.BatchNormalization(virtual_batch_size=8)
        inp = keras.layers.Input(shape=(None, None, 3))
        norm(inp)


fx = BatchNormalizationV2Test()
fx.test_basic_batchnorm_v2_none_shape_and_virtual_batch_size()
# codeEnd
