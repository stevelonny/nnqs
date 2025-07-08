import tensorflow as tf
import numpy as np

class RBM(tf.Module):

    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.a = tf.Variable(tf.random.normal(n_visible, stddev=0.01), name = 'visible_bias')
        self.b = tf.Variable(tf.random.normal(n_hidden, stddev=0.01), name = 'hidden_bias')
        self.W = tf.Variable(tf.random.normal((n_visible, n_hidden), stddev=0.01), name = 'weights')

    def log_psi(self, samples):

        sum_visible = tf.reduce_sum(self.a * samples, axis=1)
        sum_hidden = tf.reduce_sum(tf.math.log(2.0*(tf.math.cosh(self.b + tf.matmul(samples, self.W)))), axis=1)

        return sum_visible + sum_hidden
    
    def psi(self, samples):
        return tf.exp(self.log_psi(samples))

    def probability(self, samples):
        return tf.exp(2.0*tf.math.real(self.log_psi(samples)))

