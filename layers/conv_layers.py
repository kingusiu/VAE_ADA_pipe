import tensorflow as tf

# custom 1d transposed convolution that expands to 2d output for vae decoder
class Conv1DTranspose(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_sz, activation, kernel_initializer, **kwargs):
        super(Conv1DTranspose, self).__init__(**kwargs)
        self.kernel_sz = kernel_sz
        self.filters = filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.ExpandChannel = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))
        self.ConvTranspose = tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=(self.kernel_sz,1), activation=self.activation, kernel_initializer=self.kernel_initializer)
        self.SqueezeChannel = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))

    def call(self, inputs):
        # expand input and kernel to 2D
        x = self.ExpandChannel(inputs) # [ B x 98 x 4 ] -> [ B x 98 x 1 x 4 ]
        # call Conv2DTranspose
        x = self.ConvTranspose(x)
        # squeeze back to 1D and return
        x = self.SqueezeChannel(x)
        return x

    def get_config(self):
        config = super(Conv1DTranspose, self).get_config()
        config.update({'kernel_sz': self.kernel_sz, 'filters': self.filters, 'activation': self.activation, 'kernel_initializer': self.kernel_initializer})
        return config
