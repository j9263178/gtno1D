# Graphical Tensor Network Operator (GTNO)

<!-- [![Build Status]()]() -->

Graphical Tensor Network Operator (GTNO) is a variational ansatz for solving quantum spin systems. By representing the polynominal of local interaction terms with finite bond dimension tensor network operator, this ansatz shows promising results for representing the ground state of several spin-1/2 models with very few parameters, its ability to reflect the symmetry of the Hamitonian also paves the way for us to study the phase transition.

In this project, we performs optimization of the ansatz by energy minimization using automatic differentiation. We utilized [DominantSparseEigenAD](https://github.com/buwantaiji/DominantSparseEigenAD), which is an extension of PyTorch that handles reverse-mode automatic differentiation of dominant eigen-decomposition process. 

## Installation

Make sure you have installed:
- [DominantSparseEigenAD](https://github.com/buwantaiji/DominantSparseEigenAD)
- PyTorch 1.8+
- scipy 1.3.+

Then clone this repo directly and start with example codes.


## Examples

We provides minimum example codes for reproducing some important results in the paper, the construction of GTNO, inital state and local hamitonian for different models are all includes in the code.

- [Phase transition in the 1D traversed field ising model](example_TFIM.py). One interplation parameters is used as the order parameter to detect Ising symmetry breaking, by applying more GTNOs we can push the predicted criticle value to the exact one g = 1.

- [Phase transition in the 1D traversed field cluster model](example_TFCM.py). By evaluating the virtual order parameters, we showed that GTNO is able relfect the prejective representation of the SPT cluster phase. We also observed criticle behavior in the variational paramters.

- [Ground state energy of 1D Heisenberg Model](example_Heisenberg.py). Using the Heisenberg GTNO, we obtained satisfying result with very few parameters comparable to VUMPS or all parameters AD approach.


## GMPOmodel

We implement the class `GMPOmodel` which inherits from `torch.nn.module`, this object contains information and optimization process of our 1D GTNO optimization problem.

To initialize the instance we have to pass several functions for tensor construction and info about number of parameters.

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

## Known issues

    1. Depending on the initial guess, ArpackNoConvergence might appear during the optimization of Heisenberg GTNO.
