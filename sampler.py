import tensorflow as tf
import numpy as np

class MRT2:
    def __init__(self, n_sites, batch_size, n_chains, n_sweeps):
        self.n_sites = n_sites
        self.batch_size = batch_size
        self.n_chains = n_chains
        self.n_sweeps = n_sweeps

    def sample(self, wave_function):
        pass
