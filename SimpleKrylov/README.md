# SimpleKrylov

This package implements a simple sparse matrix type and a simple Lanczos algorithm.

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificComputingDemos
$ make init-SimpleKrylov   # initialize the environment in SimpleKrylov
$ make test-SimpleKrylov   # run the tests
```

## References
For professional use, please use the standard library `SparseArrays` and the [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl) package.