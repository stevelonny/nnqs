import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class MRT2:
    """
    Metropolis-Hastings sampler for RBM wave functions using TensorFlow Probability.

    This class implements a single-site Metropolis algorithm for binary spin systems.
    It supports parallel chains, burn-in, and diagnostics for acceptance rate.

    Attributes:
        n_sites (int): Number of sites (spins) in the system.
        batch_size (int): Number of samples to generate in each batch.
        n_chains (int): Number of parallel Markov chains to run.
        n_sweeps (int): Number of sweeps (MCMC steps) before returning samples.
        current_state (tf.Variable): Current state of all chains.
    """

    def __init__(self, n_sites, batch_size, n_chains, n_sweeps):
        """
        :param n_sites: number of sites in the system
        :param batch_size: number of samples to generate in each batch
        :param n_chains: number of parallel chains to run
        :param n_sweeps: number of sweeps to perform before returning samples
        """
        self.n_sites = n_sites
        self.batch_size = batch_size
        self.n_chains = n_chains
        self.n_sweeps = n_sweeps

        self.current_state = tf.Variable(
            tf.random.uniform((n_chains, n_sites), minval=0, maxval=2, dtype=tf.int32),
            trainable=False,
            name="mrt2_state",
        )

    def _single_site_flip(self, state, seed=None):
        """
        Propose a new state by flipping a single random site.
        Compatible with TFP's new_state_fn signature.
        """
        batch_size = tf.shape(state)[0]
        n_sites = self.n_sites

        sites = tf.random.uniform((batch_size,), maxval=n_sites, dtype=tf.int32)

        mask = tf.one_hot(sites, n_sites, dtype=tf.int32)

        flipped_state = state + mask * (
            tf.ones(tf.shape(state), dtype=tf.int32) - 2 * state
        )
        return flipped_state

    def sample(self, wave_function):
        """
        Generate samples using TensorFlow Probability MCMC.

        :param wave_function: RBM wave function with log_prob method
        :return: samples of shape (batch_size, n_sites)
        """

        def target_log_prob_fn(*state_parts):
            if len(state_parts) == 1:
                state = state_parts[0]
            else:
                state = state_parts[0]

            return wave_function.log_prob(state)

        kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob_fn, new_state_fn=self._single_site_flip
        )

        samples_per_chain = tf.maximum(1, self.batch_size // self.n_chains)
        # print(f"Samples per chain: {samples_per_chain.numpy()}")
        samples = tfp.mcmc.sample_chain(
            num_results=samples_per_chain,
            current_state=self.current_state,
            kernel=kernel,
            num_burnin_steps=self.n_sweeps,  # Burnin steps
            num_steps_between_results=1,
            trace_fn=None,
            parallel_iterations=self.n_chains,
        )
        self.current_state.assign(samples[-1])

        samples_flat = tf.reshape(samples, (-1, self.n_sites))
        final_samples = samples_flat[: self.batch_size]

        return tf.cast(final_samples, tf.float32)

    def get_acceptance_rate(self, kernel_results):
        """
        Calculate the acceptance rate from kernel results.

        :param kernel_results: Results from MCMC sampling
        :return: Average acceptance rate
        """
        return tf.reduce_mean(tf.cast(kernel_results.is_accepted, tf.float32))

    def sample_with_diagnostics(self, wave_function):
        """
        Generate samples and return diagnostics.

        :param wave_function: RBM wave function with log_prob method
        :return: (samples, acceptance_rate)
        """

        def target_log_prob_fn(*state_parts):
            if len(state_parts) == 1:
                state = state_parts[0]
            else:
                state = state_parts[0]

            return wave_function.log_prob(state)

        kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob_fn, new_state_fn=self._single_site_flip
        )

        samples_per_chain = tf.maximum(1, self.batch_size // self.n_chains)

        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=samples_per_chain,
            current_state=self.current_state,
            kernel=kernel,
            num_burnin_steps=self.n_sweeps,
            num_steps_between_results=1,
            trace_fn=lambda _, results: results.is_accepted,
            parallel_iterations=self.n_chains,
        )

        self.current_state.assign(samples[-1])

        acceptance_rate = tf.reduce_mean(tf.cast(kernel_results, tf.float32))

        samples_flat = tf.reshape(samples, (-1, self.n_sites))
        final_samples = samples_flat[: self.batch_size]

        return tf.cast(final_samples, tf.float32), acceptance_rate, kernel_results
