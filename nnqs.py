import tensorflow as tf
import numpy as np


class RBM(tf.Module):

    def __init__(
        self, n_visible, n_hidden, std_visible=0.1, std_hidden=0.1, std_weights=0.1
    ):
        super(RBM, self).__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.std_visible = std_visible
        self.std_hidden = std_hidden
        self.std_weights = std_weights

        self.a = tf.Variable(
            tf.complex(
                tf.random.normal((n_visible,), stddev=std_visible),
                tf.random.normal((n_visible,), stddev=std_visible),
            ),
            name="visible_bias"
        )
        self.b = tf.Variable(
            tf.complex(
                tf.random.normal((n_hidden,), stddev=std_hidden),
                tf.random.normal((n_hidden,), stddev=std_hidden),
            ),
            name="hidden_bias"
        )
        self.W = tf.Variable(
            tf.complex(
                tf.random.normal((n_visible, n_hidden), stddev=std_weights),
                tf.random.normal((n_visible, n_hidden), stddev=std_weights),
            ),
            name="weights"
        )

    def log_psi(self, samples):
        casted_samples = 2.0 * tf.cast(samples, tf.complex64) - tf.ones_like(samples, dtype=tf.complex64)
        sum_visible = tf.reduce_sum(self.a * casted_samples, axis=1)
        w_h = self.b + tf.matmul(casted_samples, self.W)
        #sum_hidden = tf.reduce_sum(tf.math.softplus(w_h), axis=1)
        sum_hidden = tf.reduce_sum(tf.math.log(2.0 * (tf.math.cosh(self.b + tf.matmul(casted_samples, self.W)))), axis=1)

        return sum_visible + sum_hidden

    def psi(self, samples):
        return tf.exp(self.log_psi(samples))

    def log_prob(self, samples):
        logpsi = self.log_psi(samples)
        return 2.0 * tf.math.real(logpsi)

    def probability(self, samples):
        return tf.exp(self.log_prob(samples))
