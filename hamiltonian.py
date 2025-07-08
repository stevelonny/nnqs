import tensorflow as tf
import numpy as np

class TFIH:
    def __init__(self, L=4, J=1.0, h=1.0, pbc=True):
        super(TFIH, self).__init__()
        self.L = L
        self.n_sites = L * L
        self.J = J
        self.h = h
        self.pbc = pbc

        self.nn_pairs = self.get_nn()

    def get_nn(self):
        nn_pairs = []
    
        for i in range(self.L):
            for j in range(self.L):
                site = i * self.L + j
                
                if j < self.L - 1:
                    nn_pairs.append((site, site + 1))
                elif self.pbc:
                    nn_pairs.append((site, i * self.L))
                
                if i < self.L - 1:
                    nn_pairs.append((site, site + self.L))
                elif self.pbc:
                    nn_pairs.append((site, j))
        
        return nn_pairs

    def local_energy(self, samples, wave_function):
        # h = -J*sum_nn(s_i^z * s_j^z) - h*sum(s_i^x)
        # need nearest neighbor
        # s_i^x flip bits (s_i^z)

        samples_size = tf.shape(samples)[0]
        energies = tf.zeros(samples_size, dtype=tf.float32)

        # j interaction
        for i, j in self.nn_pairs:
            s_i = samples[:, i]
            s_j = samples[:, j]
            energies -= self.J * s_i * s_j

        # transverse field
        log_psi_original = wave_function.log_psi(samples)
        for site in range(self.n_sites):
            flipped_bits = tf.identity(samples)
            flipped_bits = tf.tensor_scatter_nd_update(
                flipped_bits, 
                [[i, site] for i in range(samples_size)],
                1 - samples[:, site] # flips
            )
            
            log_psi_flipped = wave_function.log_psi(flipped_bits)
            energies -= self.h * tf.exp(log_psi_flipped - log_psi_original)
        
        return energies            
