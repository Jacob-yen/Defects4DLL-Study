import numpy as np
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras import keras_parameterized
import pickle
from tensorflow.python.ops import array_ops
import tensorflow as tf
import json
import tensorflow

class TestTrainingWithMetrics(keras_parameterized.TestCase):
    def test_add_metric_order(self):
        class MyLayer(keras.layers.Layer):

            def call(self, inputs, training=None, mask=None):
                self.add_metric(
                    array_ops.ones([32]) * 2.0, name='two', aggregation='mean')
                return inputs

        class MyModel(keras.Model):

            def __init__(self, **kwargs):
                super(MyModel, self).__init__(**kwargs)
                self._sampler = MyLayer(name='sampler')

            def call(self, inputs, training=None, mask=None):
                z = self._sampler(inputs)
                self.add_metric(
                    array_ops.ones([32]) * 1.0, name='one', aggregation='mean')
                self.add_metric(
                    array_ops.ones([32]) * 3.0, name='three', aggregation='mean')
                return z

        np.random.seed(100)
        xdata = np.random.uniform(size=[32, 16]).astype(np.float)
        dataset_train = dataset_ops.Dataset.from_tensor_slices((xdata, xdata))
        dataset_train = dataset_train.batch(32, drop_remainder=True)

        model = MyModel()
        model.compile(
            optimizer='sgd',
            loss='mse')
        history = model.fit(dataset_train, epochs=3)
        self.assertDictEqual(history.history, {'loss': [0.0, 0.0, 0.0], 'three': [3.0, 3.0, 3.0], 'two': [2.0, 2.0, 2.0], 'one': [1.0, 1.0, 1.0]})


fx = TestTrainingWithMetrics()
fx.test_add_metric_order()
