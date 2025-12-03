import tensorflow as tf
import tensorflow

@tf.function
def kron_error(a, b):
    return tf.experimental.numpy.kron(a, b)


a = tf.constant([[1., 2], [3, 4]], dtype=tf.float64)
b = tf.constant([[1., 2., 3., ], [-1, 4, 5]], dtype=tf.float64)
kron_error(a, b)
