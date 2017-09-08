import tensorflow as tf
from tensorflow.python.ops import init_ops
import pickle
import os

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


def print_log(experiment_name):
    log_data = load_from_file('{}/results/{}/results.log'.format(
        os.path.join(os.path.abspath(__file__).split('/modNN')[0], 'modNN'), experiment_name))
    print('\naverage performance of model over 5 folds\n')
    for i, e in enumerate(log_data['epochs']):
        print('epoch {0:2} error_train ........ acc_train ........ error_valid ........ acc_valid ........'
              .format(e + 1))
        for output_name, stats in log_data['metrics']:
            print('         {0:11} {1:.6f}           {2:.6f}             {3:.6f}           {4:.6f}'.format(
                output_name[:11],  # up to 11 characters long
                stats['error_train'][i],
                stats['accuracy_train'][i],
                stats['error_valid'][i],
                stats['accuracy_valid'][i]))


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
                              kernel_initializer=init_ops.glorot_normal_initializer(),
                              bias_initializer=init_ops.zeros_initializer(),
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



