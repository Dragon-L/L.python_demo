import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from utils import create_batch


class Model(object):
    def __init__(self, embedding_length, num_of_tags, vocabulary_size, n_hidden_rnn, pad_token_index, pad_tag_index,
                 keep_drop=None,
                 learning_rate=None):
        self.embedding_dim = embedding_length
        self.num_of_tags = num_of_tags
        self.vocabulary_size = vocabulary_size
        self.n_hidden_rnn = n_hidden_rnn
        self.pad_token_index = pad_token_index
        self.pad_tag_index = pad_tag_index
        self.keep_drop = keep_drop
        self.learning_rate = learning_rate
        tf.logging.set_verbosity(tf.logging.INFO)
        self._declare_placeholder()
        self._build_layers()
        self._compute_loss()
        self._perform_optimize()
        self._compute_predictions()

    def _declare_placeholder(self):
        self.input_batch_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch')
        self.label_batch_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='label_batch')
        self.sequence_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')
        self.dropout_ph = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

    def _build_layers(self):
        init_embedding_matrix = np.random.rand(self.vocabulary_size, self.embedding_dim) / np.sqrt(self.embedding_dim)
        # shape: [vocabulary_size, embedding_dim]
        embedding_matrix_variable = tf.Variable(init_embedding_matrix, dtype=tf.float32)
        # shape: [batch_size, max_length, embedding_dim]
        input_embedding = tf.nn.embedding_lookup(embedding_matrix_variable, self.input_batch_ph)

        forward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.n_hidden_rnn),
                                                     input_keep_prob=self.dropout_ph,
                                                     output_keep_prob=self.dropout_ph,
                                                     state_keep_prob=self.dropout_ph)

        backward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.n_hidden_rnn),
                                                      input_keep_prob=self.dropout_ph,
                                                      output_keep_prob=self.dropout_ph,
                                                      state_keep_prob=self.dropout_ph)

        (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                                               input_embedding, self.sequence_length_ph,
                                                                               dtype=tf.float32)
        output = tf.concat([forward_output, backward_output], axis=2)
        # shape: [batch_size, max_length, num_of_tags]
        self.digits = tf.layers.dense(output, self.num_of_tags)

    def _compute_predictions(self):
        result = tf.nn.softmax(self.digits)

        # shape: [batch_size, max_length]
        self.predictions = tf.math.argmax(result, axis=-1)

    def _compute_loss(self):
        # shape: [batch_size, max_length, num_of_tags]
        labels = tf.one_hot(self.label_batch_ph, self.num_of_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(labels, self.digits)

        # shape: [batch_size, sequence_length, num_of_tags]
        mask = tf.cast(tf.not_equal(labels, self.pad_token_index), dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.multiply(mask, loss_tensor))

    def _perform_optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        gradients_and_variables = optimizer.compute_gradients(self.loss)
        threshold = tf.constant(1.0, dtype=tf.float32)
        gradients_and_variables = [(tf.clip_by_norm(gradient, threshold), variable) for gradient, variable in
                                   gradients_and_variables]
        self.train_op = optimizer.apply_gradients(gradients_and_variables)

    def train(self, x, y, batch_size, is_shuffle, epoch):
        with tf.Session() as sess:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            init_op = tf.initializers.global_variables()
            sess.run(init_op)
            for i in range(epoch):
                for x_batch, y_batch, lens in create_batch(x, y, batch_size, is_shuffle, self.pad_token_index,
                                                           self.pad_tag_index):
                    feed_dict = {
                        self.input_batch_ph: x_batch,
                        self.label_batch_ph: y_batch,
                        self.sequence_length_ph: lens,
                        self.dropout_ph: self.keep_drop,
                        self.learning_rate_ph: self.learning_rate
                    }
                    sess.run(self.train_op, feed_dict=feed_dict)
