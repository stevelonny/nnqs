import tensorflow as tf
import numpy as np

class TFIH:
    def __init__(self, n_sites, J=1.0, h=1.0, pbc=True):
        super(TFIH, self).__init__()
        self.n_sites = n_sites
        self.J = J
        self.h = h
        self.pbc = pbc

    def local_energy(self, samples, wave_function):
        pass
