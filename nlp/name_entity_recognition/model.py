import numpy as np
import tensorflow as tf

from evaluation import precision_recall_f1
from utils import create_batch, create_single_batch


class Model(object):
    def __init__(self, embedding_length, num_of_tags, vocabulary_size, n_hidden_rnn, pad_token_index, pad_tag_index,
                 idx2tag,
                 keep_drop,
                 learning_rate,
                 learning_rate_decay):
        self.embedding_dim = embedding_length
        self.num_of_tags = num_of_tags
        self.vocabulary_size = vocabulary_size
        self.n_hidden_rnn = n_hidden_rnn
        self.pad_token_index = pad_token_index
        self.pad_tag_index = pad_tag_index
        self.idx2tag = idx2tag
        self.keep_drop = keep_drop
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
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
        # shape: [batch_size, max_length]
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(labels, self.digits, name='loss_tensor')

        # shape: [batch_size, sequence_length]
        mask = tf.cast(tf.not_equal(self.input_batch_ph, self.pad_token_index), dtype=tf.float32)
        # Use mask to wipe out pad loss
        self.loss = tf.reduce_mean(tf.multiply(mask, loss_tensor))

    def _perform_optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        gradients_and_variables = optimizer.compute_gradients(self.loss)
        threshold = tf.constant(1.0, dtype=tf.float32)
        gradients_and_variables = [(tf.clip_by_norm(gradient, threshold), variable) for gradient, variable in
                                   gradients_and_variables]
        self.train_op = optimizer.apply_gradients(gradients_and_variables)

    def _predict_for_batch(self, x_batch, lens, session):
        feed_dit = {
            self.input_batch_ph: x_batch,
            self.sequence_length_ph: lens,
            self.dropout_ph: self.keep_drop
        }
        return session.run(self.predictions, feed_dict=feed_dit)

    def train(self, x, y, batch_size, is_shuffle, epoch, x_val, y_val, x_test, y_test):
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            init_op = tf.initializers.global_variables()
            sess.run(init_op)

            y_tags = [self.idx2tag[y_index] for y_sequence in y for y_index in y_sequence]
            print('-' * 5 + 'Validation set quality' + '-' * 5)
            self._eval(sess, x, y_tags)
            print('-' * 5 + 'Train set quality' + '-' * 5)
            self._eval(sess, x_val, y_val)
            for i in range(epoch):
                print('start to epoch: %i' % (i + 1))
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
                self.learning_rate = self.learning_rate / self.learning_rate_decay
                print('-' * 5 + 'Validation set quality' + '-' * 5)
                self._eval(sess, x, y_tags)
                print('-' * 5 + 'Train set quality' + '-' * 5)
                self._eval(sess, x_val, y_val)

            print('-' * 5 + 'Test set quality' + '-' * 5)
            self._eval(sess, x_test, y_test)

    def _eval(self, sess, x, y):
        x, lens = create_single_batch(x, self.pad_token_index)
        predictions = self._predict_for_batch(x, lens, sess)
        y_pred = []
        for token, pred in zip(x, predictions):
            pad_indexs = np.where(token == self.pad_token_index)[0]
            if pad_indexs.size != 0:
                first_pad_index = pad_indexs[0]
                filtered_pred_indexs = pred[:first_pad_index]
            else:
                filtered_pred_indexs = pred
            filtered_pred_tag = [self.idx2tag[pred_index] for pred_index in filtered_pred_indexs]
            y_pred.extend(filtered_pred_tag)

        assert len(y_pred) == len(y)
        precision_recall_f1(y, y_pred, short_report=True)
