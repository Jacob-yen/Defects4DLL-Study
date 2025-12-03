import tensorflow
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.utils import np_utils
import numpy as np
from tensorflow.python.training.rmsprop import RMSPropOptimizer
import tensorflow as tf
import pickle
class LossWeightingTest(keras_parameterized.TestCase):
    def test_class_weights(self):
        num_classes = 5
        batch_size = 5
        epochs = 10
        weighted_class = 3
        weight = .5
        train_samples = 1000
        test_samples = 1000
        input_dim = 5
        learning_rate = 0.001
        model = testing_utils.get_small_sequential_mlp(
            num_hidden=10, num_classes=num_classes, input_dim=input_dim)
        model.compile(
            loss='categorical_crossentropy',
            metrics=['acc', metrics_module.CategoricalAccuracy()],
            weighted_metrics=['mae', metrics_module.CategoricalAccuracy()],
            optimizer=RMSPropOptimizer(learning_rate=learning_rate))
        np.random.seed(1337)
        (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
            train_samples=train_samples,
            test_samples=test_samples,
            input_shape=(input_dim,),
            num_classes=num_classes)
        int_y_test = y_test.copy()
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
        test_ids = np.where(int_y_test == np.array(weighted_class))[0]
        class_weight = dict([(i, 1.) for i in range(num_classes)])
        class_weight[weighted_class] = weight
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs // 3,
            verbose=0,
            class_weight=class_weight,
            validation_data=(x_train, y_train))
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs // 2,
            verbose=0,
            class_weight=class_weight)
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs // 2,
            verbose=0,
            class_weight=class_weight,
            validation_split=0.1)
        model.train_on_batch(
            x_train[:batch_size], y_train[:batch_size], class_weight=class_weight)
        ref_score = model.evaluate(x_test, y_test, verbose=0)
        score = model.evaluate(
            x_test[test_ids, :], y_test[test_ids, :], verbose=0)
        self.assertEqual(score[0] < ref_score[0], True)


fx = LossWeightingTest()
fx.test_class_weights()
# codeEnd
