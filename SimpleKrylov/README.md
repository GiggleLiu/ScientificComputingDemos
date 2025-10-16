# SimpleKrylov

This demo implements Krylov subspace methods for computing eigenvalues of large matrices, including the Lanczos algorithm for symmetric matrices and the Arnoldi iteration for general matrices. The main reference is: https://book.jinguo-group.science/stable/chap5/krylov/

## Contents
- Lanczos algorithm with reorthogonalization (for symmetric matrices)
- Arnoldi iteration (for general matrices)
- Householder transformations for numerical stability

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificComputingDemos
$ dir=SimpleKrylov make init   # initialize the environment in SimpleKrylov and SimpleKrylov/examples
$ dir=SimpleKrylov make example   # run the script SimpleKrylov/examples/main.jl
```

## References
For professional use, please use [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl).