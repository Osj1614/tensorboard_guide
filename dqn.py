import tensorflow as tf
import numpy as np

class dqn:
    def __init__(self, sess, input, network, epsilon=0.1, gamma=0.99, learning_rate=0.0001, name="dqn"):
        self.input = input
        self.sess = sess
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.name = name
        self.bulid_network(network)

    def bulid_network(self, network):
        with tf.variable_scope("dqn"):
            self.actions = tf.placeholder(tf.int32, [None], name="actions")
            self.value_expect = tf.placeholder(tf.float32, [None], name="value_expect")
            self.q_value = network
            with tf.variable_scope("loss"):
                self.loss = tf.reduce_mean(tf.squared_difference(tf.reduce_sum(self.q_value * tf.one_hot(self.actions, self.q_value.shape[1]), axis=1), self.value_expect))
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def get_value(self, input):
        return self.sess.run(self.q_value, feed_dict={self.input : input})

    def get_action(self, input):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.q_value.shape[1])
        return np.argmax(self.get_value([input]))

    def train_batch(self, s_lst, a_lst, r_lst, done_lst, summary=None):
        s2_lst = s_lst[1:]
        s_lst = s_lst[:-1]
        expect_lst = r_lst + self.gamma * np.max(self.get_value(s2_lst), axis=1) * done_lst
        self.sess.run(self.train, feed_dict={self.input:s_lst, self.actions:a_lst, self.value_expect:expect_lst})
        if summary != None:
            return self.sess.run(summary, feed_dict={self.input:s_lst, self.actions:a_lst, self.value_expect:expect_lst})