# ImageProcessing

This demo implements two methods for image compression, the Singular Value Decomposition (SVD) and the Fast Fourier Transform (FFT). The main reference is: https://book.jinguo-group.science/stable/chap3/fft/

## Contents
- Fast Fourier Transform (FFT)
- Singular Value Decomposition (SVD)
- Image processing toolkit: [Images.jl](https://github.com/JuliaImages/Images.jl)

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificCompuingDemos
$ make init-ImageProcessing   # initialize the environment in ImageProcessing and ImageProcessing/examples
$ make example-ImageProcessing   # run the script ImageProcessing/examples/main.jl
```