import tensorflow as tf
import numpy as np


class RBM(tf.Module):

    def __init__(
        self, n_visible=4, n_hidden=8, std_visible=0.1, std_hidden=0.1, std_weights=0.1, path=None
    ):
        super(RBM, self).__init__()
        if path is not None:
            param = np.load(path)
            self.n_visible = param['n_visible']
            self.n_hidden = param['n_hidden']
            self.a = tf.Variable(param['a'], name="visible_bias")
            self.b = tf.Variable(param['b'], name="hidden_bias")
            self.W = tf.Variable(param['W'], name="weights")
        else:
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

    @tf.function
    def log_psi(self, samples):
        casted_samples = 2.0 * tf.cast(samples, tf.complex64) - tf.ones_like(samples, dtype=tf.complex64)
        sum_visible = tf.reduce_sum(self.a * casted_samples, axis=1)
        w_h = self.b + tf.matmul(casted_samples, self.W)
        #sum_hidden = tf.reduce_sum(tf.math.softplus(w_h), axis=1)
        sum_hidden = tf.reduce_sum(tf.math.log(2.0 * (tf.math.cosh(self.b + tf.matmul(casted_samples, self.W)))), axis=1)

        return sum_visible + sum_hidden

    def log_prob(self, samples):
        logpsi = self.log_psi(samples)
        return 2.0 * tf.math.real(logpsi)
    
    def save_parameters(self, path):
        np.savez(
            path,
            n_visible=self.n_visible,
            n_hidden=self.n_hidden,
            a=self.a.numpy(),
            b=self.b.numpy(),
            W=self.W.numpy()
        )
    
    def load_parameters(self, path):
        data = np.load(path)
        self.n_visible = data['n_visible']
        self.n_hidden = data['n_hidden']
        self.a.assign(data['a'])
        self.b.assign(data['b'])
        self.W.assign(data['W'])

