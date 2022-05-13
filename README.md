# Generic Tensor Network Operator (GTNO)

<!-- [![Build Status]()]() -->

Generic Tensor Network Operator (GTNO) combines the insight of imaginary time evolution and variational optimization for studying the quantum many-body spin systems. The central idea of GTNO is to construct several operators generated from the Hamiltonian and lift the weight of different operators as the variational parameters, this ansatz results in promising ground states and symmetries of the local tensor characterizing the quantum phases.

In this repository we provide minimum example codes for the construction of GTNO for several 1D models, we implement the `GMPOmodel()` class for applying GTNO to cutomized initial states and to variantionally obtain the ground states via automatic differentiation (AD). Several observable measuments for probing the quantum phase are also included.

## Installation

Make sure you have installed:
- [DominantSparseEigenAD](https://github.com/buwantaiji/DominantSparseEigenAD)
- PyTorch 1.8+
- scipy 1.3.+

Then clone this repo directly and start with example codes.


## Examples


- [Phase transition in the 1D traversed field ising model](example_TFIM.py). Using the virtual order parameters and entanglement order parameters, we probe the phase transition between the polarized and spontaneously symmetry-broken phases using GTNO.

- [Phase transition in the 1D traversed field cluster model](example_TFCM.py). Using the virtual order parameters, we showed that GTNO is able relfect the prejective representation of the SPT cluster phase.

- [Ground state energy of 1D Heisenberg Model](example_Heisenberg.py). Using the Heisenberg GTNO, we obtained satisfying result with very few parameters comparable to VUMPS.


## GMPOmodel

We implement the class `GMPOmodel()` which inherits from `torch.nn.module`, this object contains information and optimization process of our 1D GTNO optimization problem.

To initialize the instance we have to pass several functions for tensor construction and info about number of parameters.

`GMPOmodel(localh, gmpo, A, numG, Aparas, Gparas)`

**Arguments**:

- **localh**: Function that returns a two-sites hamitonian `torch.Tensor` with shape `(d, d, d, d)`.
- **gmpo**: Function that returns a GMPO `torch.Tensor` with shape `(d, d, D, D)`.
- **A**: Function that returns a state that GMPO will apply to, should be `torch.Tensor` with shape `(d)` or `(d, D, D)`.
- **numG**: Number of times we apply GMPO to A.
- **Aparas**: Number of parameters in A.
- **Gparas**: Number of parameters in single GMPO.

Apart from methods in `torch.nn.module`, `GMPOmodel` have some methods which help to reduce the coding work when studying the problem.

**Methods**:

- **.setcs(cs = `torch.Tensor`)**: Assign the parameters in the ansatz with specific values, the argument should be a tensor with shape `(Aparas + numG * Gparas,)`.

- **.setreqgrad(reqgrad = `torch.Tensor`)**: Assign a vector describe the `req_grad` value of parameters, the argument should be a tensor contains either `0`( `req_grad` will be `False` ) or `1` ( `req_grad` will be `True` ) with shape `(Aparas + numG * Gparas,)`.

- **.getcsarray()**: Return a copy tensor `torch.Tensor` of the current paramters in the `GMPOmodel`.

<!-- ## Known issues

    1. Depending on the initial guess, ArpackNoConvergence might appear during the optimization of Heisenberg GTNO. -->
