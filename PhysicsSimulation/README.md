# PhysicsSimulation


This demo implements the physics simulation of spring-mass systems and planet orbits. The main tool we are using is the geometric numerical integration method. The main reference is the book by Hairer[^Hairer2006].

## Contents
- Spring mass system ([examples/spring_sample.jl](examples/spring_sample.jl) and [examples/spring.jl](examples/spring.jl))
- Planet orbits ([examples/planets.jl](examples/planets.jl))
- Automatic differentiation ([examples/planets.jl](examples/planets.jl))

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificCompuingDemos
$ make init-PhysicsSimulation   # initialize the environment in PhysicsSimulation and PhysicsSimulation/examples
$ make example-PhysicsSimulation   # run the script PhysicsSimulation/examples/main.jl
```

## References
[^Hairer2006]: Hairer, Ernst, et al. "Geometric numerical integration." Oberwolfach Reports 3.1 (2006): 805-882.
