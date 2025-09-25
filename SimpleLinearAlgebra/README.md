# SimpleLinearAlgebra

Implement fundamental linear algebra operations from scratch. This educational package demonstrates how common linear algebra algorithms work under the hood, with clear implementations that prioritize readability over performance.
The main reference is the book "Matrix Computations" by Golub[^Golub2016].

## Features

This package implements the following linear algebra operations:

- **Matrix Factorizations**
  - LU Decomposition (with and without pivoting)
  - QR Decomposition (using Householder reflections and Givens rotations)
  - Gram-Schmidt Orthogonalization (classical and modified)

- **Linear System Solvers**
  - Forward and backward substitution
  - Linear system solving via LU factorization

- **Fast Algorithms**
  - Strassen's matrix multiplication algorithm
  - Fast Fourier Transform (FFT) and Inverse FFT

## Run examples

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificComputingDemos
$ dir=SimpleLinearAlgebra make init   # initialize the environment in SimpleLinearAlgebra and SimpleLinearAlgebra/examples
$ dir=SimpleLinearAlgebra make test   # run the tests in SimpleLinearAlgebra/test
$ dir=SimpleLinearAlgebra make example   # run the script SimpleLinearAlgebra/examples/main.jl
```

## References
[^Golub2016]: Golub, G.H., 2016. Matrix Computation 25, 228–234. https://doi.org/10.4037/ajcc2016979
