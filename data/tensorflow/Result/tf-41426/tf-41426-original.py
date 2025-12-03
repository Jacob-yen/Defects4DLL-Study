import tensorflow
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import input_layer as input_layer_lib
import numpy as np


class AddLossTest(keras_parameterized.TestCase):

    def test_add_loss_crossentropy_backtracking(self):
        inputs = input_layer_lib.Input((2,))
        labels = input_layer_lib.Input((1,))
        outputs = layers.Dense(1, activation='sigmoid')(inputs)
        model = functional.Functional([inputs, labels], outputs)
        model.add_loss(losses.binary_crossentropy(labels, outputs))
        model.compile('adam')
        x = np.random.random((2, 2))
        y = np.random.random((2, 1))
        model.fit([x, y])


fx = AddLossTest()
fx.test_add_loss_crossentropy_backtracking()
# codeEnd
