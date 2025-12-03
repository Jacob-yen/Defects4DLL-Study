import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
import numpy as np


class TrainingTest(keras_parameterized.TestCase):
    def test_gradients_are_none(self):
        class DenseWithExtraWeight(keras.layers.Dense):

            def build(self, input_shape):
                # Gradients w.r.t. extra_weights are None
                self.extra_weight_1 = self.add_weight('extra_weight_1', shape=(),
                                                      initializer='ones')
                super(DenseWithExtraWeight, self).build(input_shape)
                self.extra_weight_2 = self.add_weight('extra_weight_2', shape=(),
                                                      initializer='ones')

        model = keras.models.Sequential([DenseWithExtraWeight(4, input_shape=(4,))])
        # Test clipping can handle None gradients
        opt = keras.optimizer_v2.adam.Adam(clipnorm=1.0, clipvalue=1.0)
        model.compile(opt, 'mse')
        inputs = np.random.normal(size=(64, 4))
        targets = np.random.normal(size=(64, 4))
        model.fit(inputs, targets)


fx = TrainingTest()
fx.test_gradients_are_none()
# codeEnd
