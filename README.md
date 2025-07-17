# Neural Net Quantum States

Investigation into Neural Nets usage as Quantum States in Many-Body Quantum Problems.

This project is made for the Deep Learning with Applications course @ UNIMI in A.Y. 2025 - 2026 . I am presenting the following article [Giuseppe Carleo, Matthias Troyer, Solving the quantum many-body problem with artificial neural networks. Science 355, 602-606 (2017)](https://doi.org/10.1126/science.aag2302), by providing a demo on ground state search.

This project relies on the TensorFlow library.

## Quick Setup and Run

### Installation
```bash
git clone https://github.com/stevelonny/nnqs.git
cd nnqs
pip install -r requirements.txt
```

### Basic Usage
```python
import nnqs
import hamiltonian
import sampler
import optimizer

# Initialize components
wave_function = nnqs.RBM(n_visible=16, n_hidden=32)
ham = hamiltonian.Ising1D(L=16, J=1.0, h=0.0, pbc=True)
sam = sampler.GibbsSampler(n_sites=16, batch_size=500, n_sweeps=10)

# Setup optimizer with callbacks
early_stop = optimizer.EarlyStoppingVariance(patience=10)
checkpoint = optimizer.CheckpointCallback("model.npz")
opt = optimizer.StochasticReconfiguration(
    wave_function, ham, sam, 
    learning_rate=0.01, 
    callbacks=[early_stop, checkpoint]
)

# Train
results = opt.train(n_iterations=1000)
```

### Example Scripts
- `trial.py`: Basic training example
- `opt_trials.ipynb`: Jupyter notebook with optimization comparisons
- `notebooks/graphs.ipynb`: Visualization and analysis tools

## Project outline
This repository provides a **ground state search** using neural networks as quantum states. They are optimized using a variational Monte Carlo approach.

```
nnqs/
├── nnqs.py                 # RBM neural net quantum state implementation
├── hamiltonian.py          # Spin system Hamiltonian
├── optimizer.py            # VMC and SR optimizer classes
├── sampler.py              # Sampling methods (MRT2, Gibbs)
├── trial.py                # Example training script
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and usage
├── LICENSE.md              # License information
├── data/
|   └── *.npz               # Bunch of waveforms and opt's histories
├── notebooks/
│   ├── graphs.ipynb        # Graphs for visualizing data
│   ├── opt_trials.ipynb    # Tests
│   └── trials.ipynb        # Original experiments (may not work)
```

### Wave function
Wave functions are represented by neural networks as subclasses of `tf.Module`.

**RBM**

`nnqs.RBM(n_visible, n_hidden, std_visible=0.1, std_hidden=0.1, std_weights=0.1, path=None)`:

Complex-valued Restricted Boltzmann Machine with `n_visible` visible units (spins) and `n_hidden` hidden units. All parameters (biases and weights) are complex-valued and initialized with Gaussian distributions. Can load pre-trained parameters from `.npz` files using the `path` parameter.

**ComplexRBM**

Alternative complex RBM implementation with enhanced numerical stability using `tf.complex()` for parameter initialization.

### Sampler
Two sampling methods extract spin configurations from the wave function probability distribution |ψ|².

**MRT2 (Metropolis)**

`sampler.MRT2(n_sites, batch_size, n_sweeps)`: TensorFlow Probability-based Metropolis-Hastings sampler using the M(RT)² algorithm for proposal and acceptance of spin configurations.

**GibbsSampler**

`sampler.GibbsSampler(n_sites, batch_size, n_sweeps)`: Alternating Gibbs sampler that leverages RBM conditional probabilities to sample visible and hidden units in sequence. Maintains persistent state across sampling calls.

### Hamiltonian

**Ising1D**: 1D Classical Ising Model (`H = -J∑σᵢσⱼ - h∑σᵢ`)

Supports periodic boundary conditions and computes:

![Local Energy](https://latex.codecogs.com/svg.image?$$E_{loc}=\frac{\hat{H}\psi\left(\vec{\bold{\sigma}},\mathcal{W}\right)}{\psi\left(\vec{\bold{\sigma}},\mathcal{W}\right)}$$)

### Optimizer
**VMC**

`optimizer.VMC(wave_function, hamiltonian, sampler, learning_rate)`: Standard Variational Monte Carlo using energy gradient descent with covariance-based gradient estimation.

**StochasticReconfiguration**

`optimizer.StochasticReconfiguration(wave_function, hamiltonian, sampler, learning_rate, epsilon)`: Natural gradient optimizer solving:

![SR](https://latex.codecogs.com/svg.image?$$\bold{S}d\mathcal{W}=-\gamma\vec{F}$$)

where **S** is the quantum geometric tensor (covariance matrix of log-derivatives) and **F** is the force vector. Includes regularization parameter `epsilon` for numerical stability.

Both optimizers support callback mechanisms for early stopping and checkpointing.

## References
Articles:
- [Giuseppe Carleo, Matthias Troyer, Solving the quantum many-body problem with artificial neural networks. Science 355, 602-606 (2017)](https://doi.org/10.1126/science.aag2302)
- [Francesco D'Angelo, Lucas Böttcher, Learning the Ising model with generative neural networks, Phys. Rev. Research 2, 023266 (2020)](https://doi.org/10.1103/PhysRevResearch.2.023266)
- [Moritz Reh, Markus Schmitt, Martin Gärttner, Optimizing design choices for neural quantum states, Phys. Rev. B 107, 195115 (2023)](https://doi.org/10.1103/PhysRevB.107.195115)


----

### License
This project is licensed under the [MIT License](LICENSE.md)
