import tensorflow as tf
import pickle
__all__ = ['save_to_file', 'load_from_file', 'variable_summaries', 'FC_layer', 'recurrent_cell']


# save/load functions that can easily be changed to different module backends
def save_to_file(data, filepath):
    filepath += '.pkl'
    with open(filepath, 'w') as f:
        pickle.dump(data, f)


def load_from_file(filepath):
    filepath += '.pkl'
    with open(filepath, 'r') as f:
        data = pickle.load(f)

    return data


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization)
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections=['train'])
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, collections=['train'])
        tf.summary.scalar('max', tf.reduce_max(var), collections=['train'])
        tf.summary.scalar('min', tf.reduce_min(var), collections=['train'])
        tf.summary.histogram('histogram', var, collections=['train'])


def FC_layer(inputs, output_dim, nonlinearity=tf.nn.relu,
             is_training=None, dropout_prob=0., name='FC_layer'):

    if dropout_prob > 0.:
        inputs = tf.layers.dropout(inputs, dropout_prob, training=is_training)

    outputs = tf.layers.dense(inputs, output_dim, nonlinearity,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              bias_initializer=tf.zeros_initializer(),
                              name=name)

    return outputs


def recurrent_cell(cell_type, output_dim,
                   is_training=None, dropout_prob=0., name='RNN_cell'):

    with tf.variable_scope(name):
        with tf.name_scope('basic_cell'):
            cell = cell_type(num_units=output_dim)

        with tf.name_scope('dropout_cell'):
            dropout_enabled = tf.logical_and(is_training, tf.greater(dropout_prob, 0.))
            keep_prob = tf.cond(dropout_enabled, lambda: 1. - tf.constant(dropout_prob), lambda: tf.constant(1.))
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)

    return cell



