# IsingModel

Solving the Ferromagnetic Ising model using the Monte Carlo method, including a Swendsen-Wang algorithms that implements cluster updates.

## Contents
- Ferromagnetic Ising model, simple Monte Carlo method
- Ferromagnetic Ising model, Swendsen-Wang algorithm

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificComputingDemos
$ make init-IsingModel   # initialize the environment in IsingModel and IsingModel/examples
$ make example-IsingModel   # run the script IsingModel/examples/main.jl
```


## References
The main reference is the Computational Physics (PY502) course at BU. The course is taught by Prof. Anders Sandvik. The course material is available at [https://physics.bu.edu/~py502/](https://physics.bu.edu/~py502/).

This demo is based on the following lecture notes:
https://physics.bu.edu/~py502/lectures5/mc.pdf