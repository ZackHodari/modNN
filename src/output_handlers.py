import tensorflow as tf
import numpy as np


class _OutputHandler(object):
    """
    Class which contains all attributes specific to an output type
    """

    @property
    def output_dim(self):
        raise NotImplementedError('Output dimension not defined')

    def init_placeholders(self):
        raise NotImplementedError('Output handler placeholders not defined')

    def feed_dict(self, batch):
        raise NotImplementedError('Target feed_dict constructor not defined')

    @property
    def predictions(self):
        return self.outputs

    # Abstract method to create error metric
    @property
    def error(self):
        raise NotImplementedError('Error metric not defined')

    # Abstract method to create accuracy metric
    @property
    def accuracy(self):
        raise NotImplementedError('Accuracy metric not defined')

    @property
    def data_config(self):
        return {}


class WaveClasses(_OutputHandler):
    @property
    def output_dim(self):
        return 3

    def init_placeholders(self):
        self.targets = tf.placeholder(tf.int32, [None, self.output_dim], name='targets')

    def feed_dict(self, batch):
        # one-hot encode targets
        targets = np.zeros((len(batch), self.output_dim))
        targets[range(len(batch)), batch.id] = 1
        return {self.targets: targets}

    @property
    def predictions(self):
        return tf.nn.softmax(self.outputs)

    @property
    def error(self):
        # calculate the softmax cross entropy loss, this limits self.outputs to sum to 1
        return tf.losses.softmax_cross_entropy(self.targets, self.outputs)

    @property
    def accuracy(self):
        # calculate the number of samples which the label with the highest probability was correct
        return tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.targets, 1), tf.argmax(self.outputs, 1)),
            tf.float32), name='accuracy')



