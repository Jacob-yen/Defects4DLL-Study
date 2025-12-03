import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
import tensorflow as tf
import pickle
import tensorflow

class MostRecentlyModifiedFileMatchingPatternTest(test.TestCase):
    def test_callback_params_samples(self):
        np.random.seed(100)
        x, y = np.ones((64, 3)), np.ones((64, 2))
        model = testing_utils.get_small_sequential_mlp(
            num_hidden=10, num_classes=2, input_dim=3)
        model.compile('sgd', 'mse')
        callback = keras.callbacks.Callback()
        model.evaluate(x, y, callbacks=[callback])
        self.assertEqual(callback.params['samples'], 64)


fx = MostRecentlyModifiedFileMatchingPatternTest()
fx.test_callback_params_samples()
# codeEnd
