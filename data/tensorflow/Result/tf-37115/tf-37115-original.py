import tensorflow as tf

keras = tf.keras


def testPreserveAspectRatioMultipleImages():
    eager = False
    test_nr = 1

    tf.config.experimental_run_functions_eagerly(eager)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    inputs = tf.constant(100., shape=[1, 100, 100, 20])
    inputs_small = tf.constant(100., shape=[1, 80, 80, 20])
    scaled_shared = Scaled1()
    test_scaled_shared = test(scaled_shared, optimizer, training=True)
    test_scaled_shared([inputs, inputs_small])


def test(op, optimizer, **kwargs):
    def run(inputs):
        with tf.GradientTape() as tape:
            tape.watch(op.trainable_variables)
            outputs = op(inputs, **kwargs)
        g = tape.gradient(outputs, op.trainable_variables)
        optimizer.apply_gradients(zip(g, op.trainable_variables))
        return outputs, g

    return run


class Scale(keras.layers.Layer):
    def __init__(self, destination_channel=None, name="Scale", **kwargs):
        super().__init__(name=name, **kwargs)
        self.destination_channel = destination_channel

    def build(self, input_shape):
        if self.destination_channel is None:
            self.destination_channel = input_shape[-1]
        self.compress_input = keras.layers.Convolution2D(int(input_shape[-1] / 2), kernel_size=1, padding='SAME',
                                                         activation=tf.nn.leaky_relu,
                                                         kernel_initializer=tf.initializers.he_normal(),
                                                         bias_initializer=tf.initializers.he_uniform())
        self.conv = keras.layers.Convolution2D(input_shape[-1], kernel_size=3, padding='SAME',
                                               activation=tf.nn.leaky_relu,
                                               kernel_initializer=tf.initializers.he_normal(),
                                               bias_initializer=tf.initializers.he_uniform())
        self.pool = keras.layers.MaxPool2D(pool_size=3, strides=1, padding="SAME")
        self.compress_output = keras.layers.Convolution2D(self.destination_channel, kernel_size=1, padding='SAME',
                                                          activation=tf.nn.leaky_relu,
                                                          kernel_initializer=tf.initializers.he_normal(),
                                                          bias_initializer=tf.initializers.he_uniform())
        super().build(input_shape)

    @tf.function
    def call(self, inputs, destination_size):
        def resize_with_aspect_ratio(image):
            return tf.image.resize(image, destination_size, preserve_aspect_ratio=True, antialias=True)
        compressed_input = self.compress_input(inputs)
        conv = self.conv(compressed_input)
        pool = self.pool(inputs)

        scaled_conv = resize_with_aspect_ratio(conv)
        scaled_pool = resize_with_aspect_ratio(pool)

        concat = keras.layers.concatenate([scaled_pool, scaled_conv])
        compressed_output = self.compress_output(concat)
        return compressed_output

    def get_config(self):
        config = super().get_config()
        config.update({'destination_channel': self.destination_channel,
                       })
        return config


class Scaled1(keras.layers.Layer):
    def __init__(self, name="Scaled1", **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        res_shape, shc_shape = input_shape
        self.scale_up = Scale(destination_channel=res_shape[-1])
        self.scale_down = Scale()
        super().build(input_shape)

    def call(self, inputs):
        inputs_res, inputs_shc = inputs
        shape1 = tf.shape(inputs_shc)[1:3]
        shape2 = tf.shape(inputs_shc)[1:3]

        scaled_res = self.scale_down(inputs_res, shape1)
        scaled_dense = self.scale_up(scaled_res, shape2)
        return scaled_dense


testPreserveAspectRatioMultipleImages()
