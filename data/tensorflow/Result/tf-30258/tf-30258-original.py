from absl.testing import parameterized
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.layers import core
import tensorflow

class PrintTrainingInfoTest(parameterized.TestCase):
    def test_dict_validation_input(self):
        train_input_0 = np.random.rand(1000, 1)
        train_input_1 = np.random.rand(1000, 1)
        train_labels = np.random.rand(1000, 1)
        val_input_0 = np.random.rand(1000, 1)
        val_input_1 = np.random.rand(1000, 1)
        val_labels = np.random.rand(1000, 1)

        input_0 = keras.Input(shape=(None,), name='input_0')
        input_1 = keras.Input(shape=(None,), name='input_1')

        class my_model(keras.Model):
            def __init__(self):
                super(my_model, self).__init__(self)
                self.hidden_layer_0 = keras.layers.Dense(100, activation="relu")
                self.hidden_layer_1 = keras.layers.Dense(100, activation="relu")
                self.concat = keras.layers.Concatenate()
                self.out_layer = keras.layers.Dense(1, activation="sigmoid")

            def call(self, inputs=[input_0, input_1]):
                activation_0 = self.hidden_layer_0(inputs['input_0'])
                activation_1 = self.hidden_layer_1(inputs['input_1'])
                concat = self.concat([activation_0, activation_1])
                return self.out_layer(concat)

        model = my_model()
        model.compile(loss="mae", optimizer="adam")

        model.fit(
            x={'input_0': train_input_0, 'input_1': train_input_1},
            y=train_labels,
            validation_data=(
                {'input_0': val_input_0, 'input_1': val_input_1}, val_labels))


fx = PrintTrainingInfoTest()
fx.test_dict_validation_input()
