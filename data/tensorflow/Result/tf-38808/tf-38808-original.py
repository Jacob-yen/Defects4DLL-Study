import tensorflow
from tensorflow.python.keras import keras_parameterized
import tensorflow as tf
import numpy as np
import pickle
class TestMetricsCorrectnessMultiIO(keras_parameterized.TestCase):

    def test(self):
        np.random.seed(100)
        X = tf.constant([[1],
                         [2],
                         [3]], dtype=tf.float32)
        y = tf.constant([[5],
                         [4],
                         [6]], dtype=tf.float32)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_dim=1, kernel_initializer='ones', bias_initializer='zeros')])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        batch_1 = model.evaluate(X, y, batch_size=1, verbose=0)
        batch_2 = model.evaluate(X, y, batch_size=2, verbose=0)
        self.assertAllClose(batch_1, batch_2)


fx = TestMetricsCorrectnessMultiIO()
fx.test()
# codeEnd
