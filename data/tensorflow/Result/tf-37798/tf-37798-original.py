import tensorflow
from tensorflow.keras.layers import Dense
import tensorflow as tf


# Define very simple classification model
class Model(tf.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.d1 = Dense(2, activation='relu')
        self.d2 = Dense(2, activation='softmax')

    @tf.function
    def __call__(self, x):
        print("Tracing the model")
        x = self.d1(x)
        return self.d2(x)


model = Model()

example_data = tf.constant([[1.0, 2.0]])
preds = model(example_data)
tf.print(preds)

# Save the model
tf.saved_model.save(model, './model_example')

# Load the saved model and convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('./model_example')
converter.convert()
