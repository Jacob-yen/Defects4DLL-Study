import tensorflow
from sklearn.datasets import load_linnerud
import tensorflow as tf
from tensorflow.keras.models import Model


class FullyConnectedNetwork(Model):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()

    def __call__(self, x, *args, **kwargs):
        return x


@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = loss_object(outputs, targets)
        train_loss(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


tf.keras.backend.set_floatx('float64')

X, y = load_linnerud(return_X_y=True)

data = tf.data.Dataset.from_tensor_slices((X, y)). \
    map(lambda a, b: (tf.divide(a, tf.reduce_max(X, axis=0, keepdims=True)), b))

train_data = data.take(16).shuffle(16).batch(4)

model = FullyConnectedNetwork()

loss_object = tf.keras.losses.Huber()

train_loss = tf.keras.metrics.Mean()

optimizer = tf.keras.optimizers.Adamax()

train_loss.reset_states()

train_data_iterator = iter(train_data)
batch = next(train_data_iterator)
x, y = batch
train_step(x, y)
