# SimpleLinearAlgebra

This demo implements some simple linear algebra operations in native Julia language. The main reference is the book by Golub[^Golub2016].

## Contents
- Generic matrix-matrix multiplication
- Householder transformation ([src/householder.jl](src/householder.jl))
- LU decomposition ([src/lu_factorization.jl](src/lu_factorization.jl) and [src/lu_factorization_partialpivoting.jl](src/lu_factorization_partialpivoting.jl))
- QR decomposition ([src/qr_factorization.jl](src/qr_factorization.jl))
- Orthogonalization ([src/orthogonalization.jl](src/orthogonalization.jl))
- Forward and back substitution ([src/back_substitution.jl](src/back_substitution.jl))
- Fast Fourier Transform (FFT) ([src/fouriertransform.jl](src/fouriertransform.jl) and [src/fastfouriertransform.jl](src/fastfouriertransform.jl))

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificCompuingDemos
$ make init-SimpleLinearAlgebra   # initialize the environment in SimpleLinearAlgebra and SimpleLinearAlgebra/examples
$ make test-SimpleLinearAlgebra   # run the tests in SimpleLinearAlgebra/test
$ make example-SimpleLinearAlgebra   # run the script SimpleLinearAlgebra/examples/main.jl
```

## References
[^Golub2016]: Golub, G.H., 2016. Matrix Computation 25, 228–234. https://doi.org/10.4037/ajcc2016979