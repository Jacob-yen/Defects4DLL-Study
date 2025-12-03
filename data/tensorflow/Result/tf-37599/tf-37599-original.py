import tensorflow
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

inp = tf.keras.Input((1,))
out = tf.keras.layers.Dense(units=1)(inp)
model = tf.keras.Model(inp, out)

model.compile(optimizer=tf.optimizers.SGD(1), loss=tf.losses.mse)

for i in range(2):
    model.fit(np.zeros((64, 1)), np.zeros((64, 1)), callbacks=[tf.keras.callbacks.TensorBoard(log_dir="tmp", profile_batch=1)], )
# codeEnd
