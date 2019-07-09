import tensorflow as tf


class Model(object):
    def __init__(self, embedding_length, num_of_tags, keep_drop=None, learning_rate=None):
        self.embedding_length = embedding_length
        self.num_of_tags = num_of_tags
        self.keep_drop = keep_drop
        self.learning_rate = learning_rate

        tf.logging.set_verbosity(tf.logging.INFO)

    def declare_placeholder(self):
        self.input_sequence_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_sequence')
        self.label_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='label')
        self.sequence_length_ph = tf.placeholder(dtype=tf.int32, shape=[], name='sequence_length')
        self.keep_drop_ph = tf.placeholder(dtype=tf.float16, shape=[], name='keep_drop')

    def build_layers(self):
        embedding_matrix = tf.Variable()
        input_embedding = tf.nn.embedding_lookup(embedding_matrix, self.input_sequence_ph)

        forward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.input_sequence_ph),
                                                     input_keep_prob=self.keep_drop_ph,
                                                     output_keep_prob=self.keep_drop_ph,
                                                     state_keep_prob=self.keep_drop_ph)

        backward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.input_sequence_ph),
                                                      input_keep_prob=self.keep_drop_ph,
                                                      output_keep_prob=self.keep_drop_ph,
                                                      state_keep_prob=self.keep_drop_ph)

        (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                                               input_embedding, self.sequence_length_ph)
        output = tf.concat(forward_output, backward_output, axis=2)
        self.digits = tf.layers.dense(output, self.num_of_tags)

        result = tf.nn.softmax(self.digits)
        self.predit = tf.math.argmax(result)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.label_ph, self.digits)

    def train(self, x_batch, y_batch, epoch):
        optimizer = tf.train.Optimizer(False)
        mini_op = optimizer.minimize(self.loss)

        with tf.Session() as sess:
            init = tf.initializers.global_variables()
            sess.run(init)

            for current_epoch in epoch:
                for x, y in zip(x_batch, y_batch):
                    feed_dict = {
                        'input_sequence': x,
                        'sequence_length': self.sequence_length_ph,
                        'keep_drop': self.keep_drop
                    }
                    sess.run(mini_op, feed_dict=feed_dict)
                predictions = sess.run(self.predit)
                print(predictions)
