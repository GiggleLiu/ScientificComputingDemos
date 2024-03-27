(Work in progress) Demos projects in the book: [Scientific Computing for Physicists](https://book.jinguo-group.science/)

[![CI](https://github.com/GiggleLiu/ScientificComputingDemos/actions/workflows/CI.yml/badge.svg)](https://github.com/GiggleLiu/ScientificComputingDemos/actions/workflows/CI.yml)

## Get started
Please make sure you have Julia installed on your local machine. If not, please download and install it with [juliaup](https://github.com/JuliaLang/juliaup).

1. Clone this repository to your local machine:
   ```bash
   $ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
   ```
2. Initialize the environment first by running the following command in the terminal:
   ```bash
   $ make init-PhysicsSimulation
   ```
3. Run the demos by running the following command in the terminal:
   ```bash
   $ make test-PhysicsSimulation
   $ make example-PhysicsSimulation
   ```
   `make-test-%` is used to run the tests in the `PhysicsSimulation` directory. `make-example-%` is used to run the examples in the `PhysicsSimulation` directory. The `PhysicsSimulation` is the name of the directory where the demos are located. You can replace it with the name of the directory where the demos are located.

## Contents
1. [MyFirstPackage](MyFirstPackage/) - Lorenz attractor
2. [PhysicsSimulation](PhysicsSimulation/) - Simulate a spring system, leapfrog method, and eigenvalue problem
3. [SimpleLinearAlgebra](SimpleLinearAlgebra/) - Implement LU decomposition, QR decomposition and FFT et al.
4. [ImageProcessing](ImageProcessing/) - Image compression with FFT and SVD
5. [CompressedSensing](CompressedSensing/) - Compressed sensing for image compression
6. [KernelPCA](KernelPCA/) - Kernel method and Kernel PCA
7. [LatticeGasCA](LatticeGasCA/) - Lattice gas cellular automata
8. [SimpleTensorNetwork](SimpleTensorNetwork/) - Tensor networks for solving spin glass and inference problems
9. [GraphClustering](GraphClustering/) - Spectral clustering algorithm