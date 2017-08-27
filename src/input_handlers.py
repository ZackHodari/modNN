import tensorflow as tf
__all__ = ['Waves']


class InputHandler(object):
    """
    Class which contains all attributes specific to an input type
    """

    @property
    def input_dim(self):
        raise NotImplementedError('Input dimension not defined')

    def init_placeholders(self):
        raise NotImplementedError('Input handler placeholders not defined')

    def feed_dict(self, batch, is_training):
        raise NotImplementedError('Input feed_dict constructor not defined')

    @property
    def data_config(self):
        raise NotImplementedError('Data provider constructor not defined')


class Waves(InputHandler):
    @property
    def input_dim(self):
        return 100

    def init_placeholders(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.input_dim], name='inputs')
        self.is_training = tf.placeholder(tf.bool, None, 'is_training')

    def feed_dict(self, batch, is_training):
        return {self.inputs: batch.y, self.is_training: is_training}

    def data_config(self):
        return {'num_samples': self.input_dim}