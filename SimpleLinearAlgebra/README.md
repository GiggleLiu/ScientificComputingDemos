# SimpleLinearAlgebra

This demo implements some simple linear algebra operations in native Julia language. The main reference is the book by Golub[^Golub2016].

## Contents
- Generic matrix-matrix multiplication
- LU decomposition
- QR decomposition
- Forward and backward substitution
- Fast Fourier Transform (FFT)

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificCompuingDemos
$ make init-SimpleLinearAlgebra   # initialize the environment in SimpleLinearAlgebra and SimpleLinearAlgebra/examples
$ make example-SimpleLinearAlgebra   # run the script SimpleLinearAlgebra/examples/main.jl
```

## References
[^Golub2016]: Golub, G.H., 2016. Matrix Computation 25, 228–234. https://doi.org/10.4037/ajcc2016979