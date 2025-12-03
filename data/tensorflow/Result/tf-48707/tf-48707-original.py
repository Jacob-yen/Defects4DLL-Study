import tensorflow
from tensorflow.python import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras import keras_parameterized
import sys
import numpy as np
import unittest
from tensorflow.python.keras.testing_utils import _SubclassModel
from tensorflow.python.ops import math_ops
import tensorflow as tf
import pickle


class KerasCallbacksTest(keras_parameterized.TestCase):

    def _get_model(self, input_shape=None, additional_metrics=None):
        additional_metrics = additional_metrics or []
        layers = [
            keras.layers.Dense(3, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ]
        model = _SubclassModel(layers, name=None, input_tensor=None)
        model.compile(
            loss='mse',
            optimizer='rmsprop',
            metrics=[keras.metrics.CategoricalAccuracy(name='my_acc')] +
                    additional_metrics)
        return model

    def test_progbar_logging_with_stateful_metrics(self):
        class AddAllOnes(keras.metrics.Metric):
            def __init__(self, name='add_all_ones', **kwargs):
                super(AddAllOnes, self).__init__(name=name, **kwargs)
                self.total = self.add_weight(name='total', initializer='zeros')

            def update_state(self, y_true, y_pred, sample_weight=None):
                self.total.assign_add(
                    math_ops.cast(math_ops.reduce_sum(y_true), dtype=dtypes.float32))

            def result(self):
                return self.total

        x_train = np.array([[0, 1, 0, 1, 0, 1, 0, 1]] * 8).astype(float)
        y_train = np.array([[1, 0], [0, 0], [1, 1], [1, 0], [0, 1], [1, 0], [1, 0],
                            [0, 0]])
        expected_log = r'(.*- loss:.*- my_acc:.*- add_all_ones: 7.0000)+'

        with self.captureWritesToStream(sys.stdout) as printed:
            model = self._get_model(
                input_shape=(8,), additional_metrics=[AddAllOnes()])
            model.fit(x_train, y_train, verbose=1, batch_size=4, shuffle=False)
        self.assertRegex(printed.contents(), expected_log)


test_case = KerasCallbacksTest()
test_case.test_progbar_logging_with_stateful_metrics()
