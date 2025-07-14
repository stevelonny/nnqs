import tensorflow as tf
import numpy as np


class TFIH:
    def __init__(self, L=4, J=1.0, h=1.0, pbc=True):
        """
        Transverse Field Ising Hamiltonian in 1D
        H = -J∑_{<i,j>}σ^z_i σ^z_j - h∑_i σ^x_i
        
        Args:
            L: System size (number of spins)
            J: Coupling strength
            h: Transverse field strength
            pbc: Periodic boundary conditions (True/False)
        """
        self.L = L
        self.J = J
        self.h = h
        self.pbc = pbc
        self.n_sites = L  # Number of sites = L for 1D
        
        self.nn_pairs = self.get_nn()

    def get_nn(self):
        """Get nearest neighbor pairs for 1D chain"""
        nn_pairs = []
        
        for i in range(self.L - 1):
            nn_pairs.append((i, i + 1))
        
        if self.pbc:
            nn_pairs.append((self.L - 1, 0))
            
        return nn_pairs

    def bit_flip_operation(self, samples, samples_size, site):
        """Flip bits at the specified site for all samples"""
        flipped_bits = tf.identity(samples)
        flipped_bits = tf.tensor_scatter_nd_update(
            flipped_bits,
            [[i, site] for i in range(samples_size)],
            1 - flipped_bits[:, site],  # flips
        )
        return flipped_bits

    def local_energy(self, samples, wave_function):
        """
        Calculate local energy for each sample
        
        Args:
            samples: Batch of spin configurations (batch_size, n_sites)
            wave_function: Wave function model with log_psi method
            
        Returns:
            Tensor of local energies for each sample (batch_size,)
        """
        samples_size = tf.shape(samples)[0]
        energies = tf.zeros(samples_size, dtype=tf.complex64)
        
        spins = 2.0 * tf.cast(samples, tf.complex64) - tf.ones_like(
            samples, dtype=tf.complex64
        )

        for i, j in self.nn_pairs:
            s_i = spins[:, i]
            s_j = spins[:, j]
            energies -= self.J * s_i * s_j
        
        log_psi_original = wave_function.log_psi(samples)
        
        for site in range(self.n_sites):
            flipped_samples = self.bit_flip_operation(samples, samples_size, site)
            log_psi_flipped = wave_function.log_psi(flipped_samples)
            
            # Add -h⟨s|σ^x_i|s'⟩ ψ(s')/ψ(s) = -h * exp(log_psi_flipped - log_psi_original)
            energies -= self.h * tf.exp(log_psi_flipped - log_psi_original)
        
        return energies

class Ising1D(TFIH):
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
        super().__init__(L=L, J=J, h=h, pbc=pbc)
        self.name = "Ising1D"
        self.n_sites = L

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
        energies = tf.zeros(samples_size, dtype=tf.complex64)
        
        spins = 2.0 * tf.cast(samples, tf.complex64) - tf.ones_like(
            samples, dtype=tf.complex64
        )

        for i, j in self.nn_pairs:
            s_i = spins[:, i]
            s_j = spins[:, j]
            energies -= self.J * s_i * s_j
        
        # Add external field contribution
        energies -= self.h * tf.reduce_sum(spins, axis=1)
        
        return energies
    
class Ising2D(Ising1D):
    def __init__(self, L=4, J=1.0, h=1.0, pbc=True):
        """
        2D Ising model with external field
        H = -J∑_{<i,j>}σ_i σ_j - h∑_i σ_i

        Args:
            L: System size (number of spins along one dimension)
            J: Coupling strength
            h: External field strength
            pbc: Periodic boundary conditions (True/False)
        """
        super().__init__(L=L**2, J=J, h=h, pbc=pbc)
        self.name = "Ising2D"
        self.n_sites = L * L
        self.nn_pairs = self.get_nn_2d(L, pbc)

    def get_nn_2d(self, L, pbc):
        """Get nearest neighbor pairs for 2D square lattice"""
        nn_pairs = []
        
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                # Right neighbor
                if j < L - 1:
                    nn_pairs.append((idx, idx + 1))
                elif pbc:
                    nn_pairs.append((idx, i * L))  # Wrap around to leftmost column
                
                # Down neighbor
                if i < L - 1:
                    nn_pairs.append((idx, idx + L))
                elif pbc:
                    nn_pairs.append((idx, (j % L)))  # Wrap around to top row of the same column
        
        return nn_pairs

class TFIH2D(TFIH):
    def __init__(self, L=4, J=1.0, h=1.0, pbc=True):
        """
        2D Transverse Field Ising Hamiltonian
        H = -J∑_{<i,j>}σ^z_i σ^z_j - h∑_i σ^x_i
        
        Args:
            L: System size (number of spins along one dimension)
            J: Coupling strength
            h: Transverse field strength
            pbc: Periodic boundary conditions (True/False)
        """
        super().__init__(L=L**2, J=J, h=h, pbc=pbc)
        self.name = "TFIH2D"
        self.n_sites = L * L
        self.nn_pairs = self.get_nn_2d(L, pbc)

    def get_nn_2d(self, L, pbc):
        """Get nearest neighbor pairs for 2D square lattice"""
        nn_pairs = []
        
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                # Right neighbor
                if j < L - 1:
                    nn_pairs.append((idx, idx + 1))
                elif pbc:
                    nn_pairs.append((idx, i * L))
                # Down neighbor
                if i < L - 1:
                    nn_pairs.append((idx, idx + L))
                elif pbc:
                    nn_pairs.append((idx, j))
        return nn_pairs

