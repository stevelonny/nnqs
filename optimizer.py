import tensorflow as tf
import numpy as np

class VMC:
    def __init__(self, wave_function, hamiltonian, sampler, learning_rate):
        self.wave_function = wave_function
        self.hamiltonian = hamiltonian
        self.sampler = sampler
        self.learning_rate = learning_rate

class StocRec:
    def __init__(self, wave_function, hamiltonian, sampler, learning_rate):
        self.wave_function = wave_function
        self.hamiltonian = hamiltonian
        self.sampler = sampler
        self.learning_rate = learning_rate

