import tensorflow as tf
import numpy as np

class Ising1D:
    def __init__(self, L=4, J=1.0, h=1.0, pbc=True):
        """
        1D Ising model with external field
        H = -J∑_{<i,j>}σ_i σ_j - h∑_i σ_i

        Args:
            L: System size (number of spins)
            J: Coupling strength
            h: External field strength
            pbc: Periodic boundary conditions (True/False)
        """
        self.J = J
        self.h = h
        self.pbc = pbc
        self.n_sites = L
        
        self.nn_pairs = self.get_nn()

    def get_nn(self):
        """Get nearest neighbor pairs for 1D chain"""
        nn_pairs = []

        for i in range(self.n_sites - 1):
            nn_pairs.append((i, i + 1))
        
        if self.pbc:
            nn_pairs.append((self.n_sites - 1, 0))

        return nn_pairs

    @tf.function
    def local_energy(self, samples, wave_function):
        """
        Calculate local energy for each sample in the 1D Ising model
        
        Args:
            samples: Batch of spin configurations (batch_size, n_sites)
            wave_function: Wave function model with log_psi method
            
        Returns:
            Tensor of local energies for each sample (batch_size,)
        """
        samples_size = tf.shape(samples)[0]
        energies = tf.zeros(samples_size, dtype=tf.float32)
        
        spins = 2.0 * samples - tf.ones_like(
            samples, dtype=tf.float32
        )

        for i, j in self.nn_pairs:
            s_i = spins[:, i]
            s_j = spins[:, j]
            energies -= self.J * s_i * s_j
        
        # Add external field contribution
        energies -= self.h * tf.reduce_sum(spins, axis=1)
        
        return energies
