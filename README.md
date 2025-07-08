# Neural Net Quantum States

Investigation into Neural Nets usage as Quantum States in Many-Body Quantum Problems.

This project is made for the Deep Learning with Applications course @ UNIMI in A.Y. 2025 - 2026 . I am presenting the following article: []() with a demo on ground state search.

## Project outline
**Ground state search:** Quantum state are represented by neural nets, which will be trained with a Variational Monte Carlo approach.

### Wave function
Custom keras Model or tf.module.
- **RBM** visible bias, hidden bias, weights hidden->visible
- **...**

### Hamiltonian
Will need to compute local energy having a batch of samples and the wave function.
$$
E_{loc} = \frac{\hat{H} \psi \left( \vec{\bold{\sigma}} , \mathcal{W} \right)}{\psi \left( \vec{\bold{\sigma}} , \mathcal{W} \right)} 
$$

### Optimizer
**VMC**

Standard Variational Monte Carlo approach. Uses samples to compute local energy and subsequent gradient using mean energy as loss function to minimieze

**Stochastic Reconfiguration**
$$
\bold{S} d \mathcal{W} = - \gamma \vec{F} 
$$
where $\bold{S}$ is the quantum geometric tensor or covariance matrix of the local operators, and $\vec{F}$ is the force vector. Weights are then updated according to this imaginary time evolution of the parameters.


## References
Articles:
- Main article
- Learning the ising model
- Comparison/Optimal choice of RBM/cVAE...

Code
- jVMC
- QuantumToolbox

----

### License
This project is licensed under the [MIT License](LICENSE.md)
