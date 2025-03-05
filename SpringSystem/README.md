# SpringSystem

This demo implements the physics simulation of spring-mass systems and planet orbits. The main tool we are using is the geometric numerical integration method. The main reference is the book by Hairer[^Hairer2006].

## Contents
- Spring-mass system ([examples/main.jl](examples/main.jl)), a comparative study between the leap-frog based simulation and the exact solution with eigenmodes.

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificComputingDemos
$ make init-SpringSystem     # initialize the environment in SpringSystem and SpringSystem/examples
$ make example-SpringSystem  # run the script SpringSystem/examples/main.jl
```

## References
[^Hairer2006]: Hairer, Ernst, et al. "Geometric numerical integration." Oberwolfach Reports 3.1 (2006): 805-882.
