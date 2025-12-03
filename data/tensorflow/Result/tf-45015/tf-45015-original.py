import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf


class RaggedDense(layers.Layer):
    def __init__(self, dims):
        super(RaggedDense, self).__init__(dims)
        self.dense = layers.Dense(dims)

    def call(self, inputs):
        return tf.ragged.map_flat_values(self.dense, inputs)


class RaggedMSE(keras.losses.Loss):
    def call(self, y_true, y_pred):
        losses = tf.ragged.map_flat_values(tf.keras.losses.mse, y_true, y_pred)
        return tf.reduce_mean(losses)


def make_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(None, 2), ragged=True))
    model.add(RaggedDense(1))
    model.compile(loss=RaggedMSE())
    return model


model = make_model()
data = np.random.random(size=64)
row_splits = [
    [0, 5, 6, 8],
    [0, 2, 8],
    [0, 1, 4, 6, 8],
    [0, 4, 8],
]
x_list = []
y_list = []
accum = 0
for rs in row_splits:
    end = accum + rs[-1] * 2
    xdata = data[accum:end].reshape(-1, 2)
    rt = tf.RaggedTensor.from_row_splits(xdata, rs)
    x_list.append(rt)
    pair_sums = np.sum(xdata, axis=1, keepdims=True)
    y = tf.RaggedTensor.from_row_splits(pair_sums, rs)
    y_list.append(y)
    accum += rs[-1] * 2
x_train = tf.ragged.stack(x_list)
y_train = tf.ragged.stack(y_list)
x_train.bounding_shape(), y_train.bounding_shape()
model.fit(x_train, y_train)
# codeEnd
