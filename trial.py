from nnqs import RBM
import optimizer
import sampler
from hamiltonian import Ising1D
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np

n_spin = 8
density = 4
n_hidden = int(n_spin * density)

ham = Ising1D(L=n_spin, J=-1.0, h=0.5, pbc=True)

wave = RBM(
    path="blank_8_32.npz",
    #n_visible=n_spin,
    #n_hidden=n_hidden,
    #std_hidden=0.01,
    #std_visible=0.01,
    #std_weights=0.01
)

gibbs_sampler = sampler.GibbsSampler(
    n_visible=n_spin, n_hidden=n_hidden, k=5, batch_size=100
)

callbacks = [
    optimizer.EarlyStoppingVariance(patience=8),
    optimizer.CheckpointCallback("8spin_4_VMC.npz")
]

opt = optimizer.StochasticReconfiguration(
    wave_function=wave,
    hamiltonian=ham,
    sampler=gibbs_sampler,
    learning_rate=0.01,
    callbacks=callbacks,
    epsilon=1e-3
)

results_sr = opt.train(n_iterations=2000)

np.savez(
    "8spin_4_VMC_history.npz",
    energies=results_sr["energies"],
    variances=results_sr["variances"],
)