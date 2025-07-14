# Neural Net Quantum States

Investigation into Neural Nets usage as Quantum States in Many-Body Quantum Problems.

This project is made for the Deep Learning with Applications course @ UNIMI in A.Y. 2025 - 2026 . I am presenting the following article [Giuseppe Carleo, Matthias Troyer, Solving the quantum many-body problem with artificial neural networks. Science 355, 602-606 (2017)](https://doi.org/10.1126/science.aag2302), by providing a demo on ground state search.

This project relies on the TensorFlow library.

## Project outline
This repository provides a **ground state search** using neural networks as quantum states. They are optimized using a variational Monte Carlo approach.

### Wave function
Wave functions are represented by neural networks. As such they are implemented as a subclass of `tf.Module`

**RBM**

`nnqs.RBM(n_visible, n_hidden, std_visible=0.1, std_hidden=0.1, std_weights=0.1)`:

initialize a Restricted Boltzman Machine with `n_visible` neurons, `n_hidden` neurons with corrispective bias and weights using a normal distribution with declared standard deviation.

### Sampler
There are two sampling methods which extracts probable spin configurations from the wave function.

**RandomWalk**

Leveraging the Tensorflow Probability module, a Monte Carlo Markov Chain is initiliazed, which uses the M(RT)2 algorithm to propose and accept configurations.

**Gibbs**

With RBM conditional probabilities for each spin (and hidden ones) can be obtained, so a sampler using a double-step gibbs sampling tehcninque is provided.

### Hamiltonian

A bunch of hamiltonians describing spin systems has been implemented. Each of them provides the method to compute the local energy.

![Local Energy](https://latex.codecogs.com/svg.image?$$E_{loc}=\frac{\hat{H}\psi\left(\vec{\bold{\sigma}},\mathcal{W}\right)}{\psi\left(\vec{\bold{\sigma}},\mathcal{W}\right)}$$)

### Optimizer
**VMC**

Standard Variational Monte Carlo approach. Uses samples to compute local energy and subsequent gradient using mean energy as loss function to minimize

**Stochastic Reconfiguration**

![SR](https://latex.codecogs.com/svg.image?$$\bold{S}d\mathcal{W}=-\gamma\vec{F}$$)

where **S** is the quantum geometric tensor or covariance matrix of the local operators, and **F** is the force vector. Weights are then updated according to this equation.

## References
Articles:
- [Giuseppe Carleo, Matthias Troyer, Solving the quantum many-body problem with artificial neural networks. Science 355, 602-606 (2017)](https://doi.org/10.1126/science.aag2302)
- [Francesco D'Angelo, Lucas Böttcher, Learning the Ising model with generative neural networks, Phys. Rev. Research 2, 023266 (2020)](https://doi.org/10.1103/PhysRevResearch.2.023266)
- [Moritz Reh, Markus Schmitt, Martin Gärttner, Optimizing design choices for neural quantum states, Phys. Rev. B 107, 195115 (2023)](https://doi.org/10.1103/PhysRevB.107.195115)


----

### License
This project is licensed under the [MIT License](LICENSE.md)
