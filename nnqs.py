import tensorflow as tf
import numpy as np


class RBM(tf.Module):
    """Restricted Boltzmann Machine for quantum wave function representation.
    
    A complex-valued RBM that can be used as a variational ansatz for quantum
    many-body systems. The wave function is parameterized as:
    
    ψ(σ) = exp(∑ᵢ aᵢσᵢ + ∑ⱼ log(2cosh(bⱼ + ∑ᵢ Wᵢⱼσᵢ)))
    
    where σᵢ ∈ {0,1} are visible units (spins), and all parameters are complex-valued.
    """

    def __init__(
        self, n_visible=4, n_hidden=8, std_visible=0.1, std_hidden=0.1, std_weights=0.1, path=None
    ):
        """Initialize the RBM with complex-valued parameters.
        
        Args:
            n_visible: Number of visible units (spins/qubits).
            n_hidden: Number of hidden units.
            std_visible: Standard deviation for visible bias initialization.
            std_hidden: Standard deviation for hidden bias initialization.
            std_weights: Standard deviation for weight matrix initialization.
            path: Optional path to load pre-trained parameters from .npz file.
        """
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
        """Compute the complex log-amplitude of the wave function.
        
        Args:
            samples: Tensor of shape (batch_size, n_visible) containing binary 
                    spin configurations (0s and 1s).
        
        Returns:
            Complex tensor of shape (batch_size,) containing log ψ(σ) for each sample.
        """
        casted_samples = 2.0 * tf.cast(samples, tf.complex64) - tf.ones_like(samples, dtype=tf.complex64)
        sum_visible = tf.reduce_sum(self.a * casted_samples, axis=1)
        w_h = self.b + tf.matmul(casted_samples, self.W)
        #sum_hidden = tf.reduce_sum(tf.math.softplus(w_h), axis=1)
        sum_hidden = tf.reduce_sum(tf.math.log(2.0 * (tf.math.cosh(self.b + tf.matmul(casted_samples, self.W)))), axis=1)

        return sum_visible + sum_hidden

    def log_prob(self, samples):
        """Compute log-probability for sampling from |ψ|².
        
        Args:
            samples: Tensor of shape (batch_size, n_visible) containing binary 
                    spin configurations.
        
        Returns:
            Real tensor of shape (batch_size,) containing 2×Re[log ψ(σ)] = log|ψ(σ)|².
        """
        logpsi = self.log_psi(samples)
        return 2.0 * tf.math.real(logpsi)
    
    def save_parameters(self, path):
        """Save RBM parameters to a .npz file.
        
        Args:
            path: String path where to save the parameters.
        """
        np.savez(
            path,
            n_visible=self.n_visible,
            n_hidden=self.n_hidden,
            a=self.a.numpy(),
            b=self.b.numpy(),
            W=self.W.numpy()
        )
    
    def load_parameters(self, path):
        """Load RBM parameters from a .npz file.
        
        Args:
            path: String path to the .npz file containing saved parameters.
        """
        data = np.load(path)
        self.n_visible = data['n_visible']
        self.n_hidden = data['n_hidden']
        self.a.assign(data['a'])
        self.b.assign(data['b'])
        self.W.assign(data['W'])

