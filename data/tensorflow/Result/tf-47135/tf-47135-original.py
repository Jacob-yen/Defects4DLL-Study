import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.applications import imagenet_utils as utils


class TestImageNetUtils(keras_parameterized.TestCase):

    def test_preprocess_input_symbolic(self, mode='torch'):
        # Test image batch
        x = np.random.uniform(0, 255, (2, 10, 10, 3))
        x2 = np.transpose(x, (0, 3, 1, 2))
        inputs2 = keras.layers.Input(shape=x2.shape[1:])
        keras.layers.Lambda(lambda x: utils.preprocess_input(x, 'channels_first', mode=mode),
                            output_shape=x2.shape[1:])(inputs2)


pc = TestImageNetUtils()
pc.test_preprocess_input_symbolic()
