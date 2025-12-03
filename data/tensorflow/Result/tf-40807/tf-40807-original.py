import tensorflow
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import input_layer as input_layer_lib
import numpy as np
import tensorflow as tf
import pickle
class CacheCorrectnessTest(keras_parameterized.TestCase):

    def test_training_passed_during_construction(self):

        def _call(inputs, training):
            if training is None:
                return inputs * -1.0
            elif training:
                return inputs
            else:
                return inputs * 0.0

        class MyLayer(base_layer.Layer):

            def call(self, inputs, training=True):
                return _call(inputs, training)

        np.random.seed(100)
        my_layer = MyLayer()
        x = np.ones((1, 10))
        inputs = input_layer_lib.Input(10)
        outputs = my_layer(inputs)
        network = functional.Functional(inputs, outputs)
        self.assertAllEqual(network(x), _call(x, True))


fx = CacheCorrectnessTest()
fx.test_training_passed_during_construction()
# codeEnd
