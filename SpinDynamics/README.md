# SpinDynamics

This package provides tools for simulating the dynamics of classical spin systems using numerical integration methods. It allows researchers and students to explore magnetic phenomena at the atomistic level.

## Contents

- Landau-Lifshitz-Gilbert equation for 3D spin dynamics simulation based on the work of [^Tsai2008].
  ```math
  \frac{d\mathbf{m}_i}{dt} = \gamma \left( \mathbf{m}_i \times \mathbf{H}_i + \alpha \left( \mathbf{m}_i \times \frac{d\mathbf{m}_i}{dt} \right) \right)
  ```
  where $\mathbf{m}_i$ is the magnetization vector, $\mathbf{H}_i = \sum_{j} J_{ij} \mathbf{m}_j + \mathbf{h}_i$ is the effective field, $\gamma$ is the gyromagnetic ratio, and $\alpha$ is the damping constant.
- Simulated bifurcation for finding the energy minimum of a spin glass model on a graph $G = (V, E)$[^Goto2021].
  ```math
  V_{\rm aSB} = \sum_{i \in V} \frac{x_i^4}{4} + \frac{a}{2} x_i^2 - c_0 \sum_{(i,j) \in E} J_{ij} x_i x_j\\
  V_{\rm bSB} = \sum_{i \in V} \frac{a}{2} x_i^2 - c_0 \sum_{(i,j) \in E} J_{ij} x_i x_j\\
  V_{\rm dSB} = \sum_{i \in V} \frac{a}{2} x_i^2 - c_0 \sum_{(i,j) \in E} J_{ij} (x_i \mathrm{sign}(x_j) + x_j \mathrm{sign}(x_i))
  ```
  where $x_i$ is the spin variable, $a$ is the bifurcation parameter, $c_0$ is the coupling strength, and $J_{ij}$ is the coupling matrix. For $\mathrm{aSB}$, $a$ ramps from $1$ to $-1$, while for $\mathrm{bSB}$ and $\mathrm{dSB}$, $a$ ramps from $1$ to $0$.

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificComputingDemos
$ make init-SpinDynamics   # initialize the environment in SpinDynamics and SpinDynamics/examples
$ make example-SpinDynamics   # run the script SpinDynamics/examples/main.jl
```

## References
[^Tsai2008]: Tsai, S.-H., Landau, D.P., 2008. Spin Dynamics: An Atomistic Simulation Tool for Magnetic Systems. Computing in Science & Engineering 10, 72–79. https://doi.org/10.1109/MCSE.2008.12
[^Goto2021]: Goto, H., Endo, K., Suzuki, M., Sakai, Y., Kanao, T., Hamakawa, Y., Hidaka, R., Yamasaki, M., Tatsumura, K., 2021. High-performance combinatorial optimization based on classical mechanics. Science Advances 7, eabe7953. https://doi.org/10.1126/sciadv.abe7953
