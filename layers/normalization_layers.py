import tensorflow as tf

class StdNormalization(tf.keras.layers.Layer):
    """normalizing input feature to std Gauss (mean 0, var 1)"""
    def __init__(self, mean_x, std_x, name='Std_Normalize', **kwargs):
        kwargs.update({'name': name, 'trainable': False})
        super(StdNormalization, self).__init__(**kwargs)
        self.mean_x = mean_x
        self.std_x = std_x

    def get_config(self):
        config = super(StdNormalization, self).get_config()
        config.update({'mean_x': self.mean_x, 'std_x': self.std_x})
        return config

    def call(self, x):
        return (x - self.mean_x) / self.std_x


class StdUnnormalization(StdNormalization):
    """ rescaling feature to original domain """

    def __init__(self, mean_x, std_x, name='Un_Normalize', **kwargs):
        super(StdUnnormalization, self).__init__(mean_x=mean_x, std_x=std_x, name=name, **kwargs)

    def call(self, x):
        return (x * self.std_x) + self.mean_x


class MinMaxNormalization(tf.keras.layers.Layer):
    """normalizing input feature to std Gauss (mean 0, var 1)"""
    def __init__(self, min_x, max_x, name='MinMax_Normalize', **kwargs):
        kwargs.update({'name': name, 'trainable': False})
        super(MinMaxNormalization, self).__init__(**kwargs)
        self.min_x = min_x
        self.max_x = max_x

    def get_config(self):
        config = super(MinMaxNormalization, self).get_config()
        config.update({'min_x': self.min_x, 'max_x': self.max_x})
        return config

    def call(self, x):
        return (x - self.min_x) / (self.max_x - self.min_x)


class MinMaxUnnormalization(MinMaxNormalization):
    """ rescaling feature to original domain """

    def __init__(self, min_x, max_x, name='MinMax_Un_Normalize', **kwargs):
        super(MinMaxUnnormalization, self).__init__(min_x=min_x, max_x=max_x, name=name, **kwargs)

    def call(self, x):
        return x * (self.max_x - self.min_x) + self.min_x

