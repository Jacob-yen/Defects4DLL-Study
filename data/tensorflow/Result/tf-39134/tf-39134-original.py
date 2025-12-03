import tensorflow
from tensorflow.python.keras import backend
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

m = tf.keras.metrics.Recall()
m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
