"""Module for variational Monte Carlo (VMC) optimizers for quantum spin models.

Provides:
- QOptimzer: base class for optimization routines.
- VMC: variational Monte Carlo optimizer using SGD.
- StochasticReconfiguration: natural gradient optimizer via stochastic reconfiguration.
"""

import tensorflow as tf
import numpy as np

class Callback:
    def on_train_begin(self, logs=None): pass
    def on_step_end( self, step, logs=None): pass
    def on_train_end(  self, logs=None): pass

class EarlyStoppingVariance(Callback):
    def __init__(self, patience=5):
        """Initialize the callback with a patience parameter.

        Args:
            patience: Number of consecutive iterations with zero variance to wait before stopping.
            min_variance: Minimum variance threshold to trigger early stopping.
        """
        self.patience = patience
        self.counter = 0
        self.stopped = False

    def on_step_end(self, step, logs=None):
        """Check the variance at the end of each step and update the stopping condition.

        Args:
            step: Current iteration step.
            logs: Dictionary containing 'variance' key.
        """
        variance = logs.get('variance', None)
        if np.isclose(variance, 0.00):
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                print(f"Early stopping triggered at step {step} due to low variance ({variance:.2e}).")
                return True  # Signal to stop training
        else:
            self.counter = 0
        return False

    def on_train_begin(self, logs=None):
        """Reset the state at the beginning of training."""
        self.counter = 0
        self.stopped = False

    def on_train_end(self, logs=None):
        """Log a message if early stopping was triggered."""
        if self.stopped:
            print("Training stopped early due to low variance.")

def CheckpointCallback(path):
    """Callback to save model parameters at the end of training."""
    class Checkpoint(Callback):
        def __init__(self, path):
            self.path = path
            self.wave_function = None

        def on_train_begin(self, logs=None):
            """Set the wave function reference at the beginning of training."""
            # This will be set by the optimizer
            pass

        def on_train_end(self, logs=None):
            """Save the wave function parameters to a file."""
            if self.wave_function is None:
                print("Warning: Wave function not set in checkpoint callback.")
                return
            
            params = {
                'n_visible': self.wave_function.n_visible,
                'n_hidden': self.wave_function.n_hidden,
                'a': self.wave_function.a.numpy(),
                'b': self.wave_function.b.numpy(),
                'W': self.wave_function.W.numpy()
            }
            np.savez(self.path, **params)
            print(f"Checkpoint saved to {self.path}")

    return Checkpoint(path)


class QOptimzer:
    """Base class for quantum optimizers applying variational Monte Carlo or stochastic reconfiguration methods.

    Args:
        wave_function: Variational wave function model (e.g., RBM).
        hamiltonian: Hamiltonian object for computing local energies.
        sampler: Sampler object for generating configuration samples.
        learning_rate: Learning rate for parameter updates.

    Attributes:
        optimizer: TensorFlow optimizer instance used for parameter updates.
    """

    def __init__(self, wave_function, hamiltonian, sampler, learning_rate=0.01, callbacks=None):
        """Initialize the optimizer with the given wave function, Hamiltonian, sampler, and learning rate.

        Args:
            wave_function: Model representing the variational wave function.
            hamiltonian: Hamiltonian object for computing local energies.
            sampler: Sampler for generating samples from |ψ|^2 distribution.
            learning_rate: Float learning rate for the internal TensorFlow optimizer.
        """
        self.wave_function = wave_function
        self.hamiltonian = hamiltonian
        self.sampler = sampler
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.callbacks = callbacks or []

    def _call(self, name, *args, **kwargs):
        for cb in self.callbacks:
            # Set wave function reference for checkpoint callback
            if hasattr(cb, 'wave_function'):
                cb.wave_function = self.wave_function
            result = getattr(cb, name)(*args, **kwargs)
            # Handle early stopping
            if name == "on_step_end" and hasattr(cb, 'stopped') and cb.stopped:
                return True
        return False

    def train(self, n_iterations, verbose=True):
        """Run optimization loop for a number of iterations.

        Args:
            n_iterations: Integer number of optimization steps to perform.
            verbose: Boolean flag to print progress each iteration.

        Returns:
            Dictionary with keys 'energies' and 'variances', containing histories.
        """
        energy_history = []
        variance_history = []
        self._call("on_train_begin", logs={})
        for i in range(n_iterations):
            energy, variance = self.optimize_step()
            energy_history.append(energy)
            variance_history.append(variance)
            logs = {'iteration':i, 'energy':energy, 'variance':variance}
            should_stop = self._call("on_step_end", step=i, logs=logs)
            if verbose:
                print(
                    f"Iteration {i}: Energy = {energy:.6f}, Variance = {variance:.6f}"
                )
            if should_stop:
                break
        self._call("on_train_end", logs={"energies": energy_history, "variances": variance_history})
        return {"energies": energy_history, "variances": variance_history}
    
    def optimize_step(self):
        """Perform a single optimization step. Must be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented by subclass.

        Returns:
            Tuple (mean_energy, variance) after the optimization step.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

class VMC(QOptimzer):
    """Variational Monte Carlo optimizer using standard energy gradient descent.

    Implements optimization by computing gradients of the energy expectation value via Monte Carlo sampling.
    """

    def __init__(self, wave_function, hamiltonian, sampler, learning_rate, callbacks=None):
        super().__init__(wave_function, hamiltonian, sampler, learning_rate, callbacks)

    def compute_gradients(self, samples, local_energies):
        """Compute parameter gradients using VMC energy derivatives.

        Args:
            samples: Tensor of sampled spin configurations.
            local_energies: Tensor of local energy values for each sample.

        Returns:
            gradients: List of gradient tensors matching trainable variable shapes.
            mean_energy: Scalar mean of the provided local energies.
        """

        with tf.GradientTape() as tape:
            log_psi = self.wave_function.log_psi(samples)

        grad_log_psi = tape.jacobian(log_psi, self.wave_function.trainable_variables)

        # print("Gradients of log_psi:", [g.shape if g is not None else None for g in grad_log_psi])

        mean_energy = tf.reduce_mean(local_energies)
        gradients = []

        for grad_i in grad_log_psi:
            if grad_i is not None:
                # Covariance term: <E_loc * grad_log_psi> - <E_loc> * <grad_log_psi>
                expand_shape = tf.concat([[tf.shape(local_energies)[0]], tf.ones(tf.rank(grad_i) - 1, dtype=tf.int32)], axis=0)
                e_loc_expanded = tf.reshape(local_energies, expand_shape)
                loc_grad = e_loc_expanded * grad_i
                grad_energy = tf.reduce_mean(loc_grad, axis=0)
                grad_log_psi_mean = tf.reduce_mean(grad_i, axis=0)

                vmc_grad = 2.0 * tf.math.real(grad_energy - mean_energy * grad_log_psi_mean)
                vmc_grad = tf.cast(vmc_grad, tf.complex64)  # Ensure dtype matches variable
                gradients.append(vmc_grad)
            else:
                gradients.append(None)

        gradients = [
            tf.convert_to_tensor(g) if g is not None else None for g in gradients
        ]

        return gradients, mean_energy

    def optimize_step(self):
        """Execute one VMC optimization step: sample, compute gradients, and apply updates.

        Returns:
            mean_energy: Scalar mean energy after this step.
            variance: Scalar variance of local energy distribution.
        """

        samples = self.sampler.sample(self.wave_function)
        local_energies = self.hamiltonian.local_energy(samples, self.wave_function)

        gradients, mean_energy = self.compute_gradients(samples, local_energies)

        #self.optimizer.apply_gradients(
        #    zip(gradients, self.wave_function.trainable_variables)
        #)

        for grad, var in zip(gradients, self.wave_function.trainable_variables):
            if grad is not None:
                var.assign_sub(self.learning_rate * grad)

        variance = tf.reduce_mean(tf.square(local_energies - mean_energy))

        return mean_energy, variance

class StochasticReconfiguration(QOptimzer):
    """Natural gradient optimizer using Stochastic Reconfiguration (quantum Fisher metric).

    Approximates the quantum geometric tensor via sample covariance of score functions.
    """

    def __init__(self, wave_function, hamiltonian, sampler, learning_rate=0.01, epsilon=0.001, callbacks=None):
        super().__init__(wave_function, hamiltonian, sampler, learning_rate, callbacks=callbacks)
        self.epsilon = epsilon

    def optimize_step(self):
        """Perform one optimization step using the quantum natural gradient.

        Samples configurations, builds the covariance (QGT) matrix, regularizes, and updates parameters.

        Returns:
            mean_energy: Scalar average local energy.
            variance: Scalar variance of the local energy distribution.
        """

        samples = self.sampler.sample(self.wave_function)
        local_energies = self.hamiltonian.local_energy(samples, self.wave_function)
        n_samples = tf.shape(samples)[0]

        # compute local operators
        with tf.GradientTape() as tape:
            log_psi = self.wave_function.log_psi(samples)
        grad_log_psi = tape.jacobian(log_psi, self.wave_function.trainable_variables)

        flat_grads = []
        param_sizes = []
        for g in grad_log_psi:
            g_flat = tf.reshape(g, [n_samples, -1])
            param_sizes.append(tf.shape(g_flat)[1])
            flat_grads.append(g_flat)

        # the QGT/covariance matrix needs centered data
        O = tf.concat(flat_grads, axis=1)
        O_mean = tf.reduce_mean(O, axis=0)
        O_centered = O - O_mean

        # compute covariance matrix / QGT
        O_O = tf.matmul(O_centered, O_centered, transpose_a=True)
        norm = tf.cast(n_samples, O_centered.dtype)
        #print("just before S")
        S = O_O / norm
        #print("S shape:", S.shape)
        #print("S dtype:", S.dtype)
        # compute force vector
        mean_energy = tf.reduce_mean(local_energies)
        F_vec = (
            tf.reduce_mean(local_energies[:, None] * O, axis=0) - mean_energy * O_mean
        )
        #print("F_vec shape:", F_vec.shape)
        #print("F_vec dtype:", F_vec.dtype)
        # solve for delta
        P = tf.shape(S)[0]
        S_reg = S + self.epsilon * tf.eye(P, dtype=S.dtype)
        gradients = tf.linalg.solve(S_reg, tf.expand_dims(F_vec, 1))
        
        gradients = tf.squeeze(gradients, 1)
        gradients = tf.split(gradients, param_sizes, axis=0)
        gradients = [
            tf.reshape(d, var.shape)
            for var, d in zip(self.wave_function.trainable_variables, gradients)
        ]

        grads_vars = []
        for grad, var in zip(gradients, self.wave_function.trainable_variables):
            grad_real = tf.math.real(grad)
            if var.dtype.is_complex:
                grad_real = tf.cast(grad_real, var.dtype)
            grads_vars.append((grad_real, var))

        for grad_val, var in grads_vars:
            var.assign_sub(self.learning_rate * grad_val)

        variance = tf.reduce_mean(tf.square(local_energies - mean_energy))
        return mean_energy, variance

