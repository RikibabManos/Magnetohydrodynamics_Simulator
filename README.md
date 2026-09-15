# Magnetohydrodynamics Simulator
---
![Orszag Tang Benchmark Test](OrszagTang_512x512_grid.gif)


## Overview 

A high-performance 2D compressible Magnetohydrodynamics (MHD) Simulator its engine implemented in C++ and visulised in python (decoupled). The solver models non-linear plasma interactions, shock dynamics and their evolution over time by coupling high-order finite volume spatial discretisations witha a divergence-free field constraint and a multi-stage IMEX time integrator.


## Description
- **High-Performance 2D Ideal/Compressible MHD Engine:** Built in C++ modelling plasma interactions and shock dynamics using Finite Volume Methods
- **Divergence-Free Magnetic Field Propagation:** Utilises a staggered-grid Upwind-Constrained Transport methods (UCT) to ensure $\nabla \cdot B = 0$ down to machine precision
- **High Order Shock Capturing:** 2nd-order MUSCL spatial reconstruction coupled with an HLL Riemann flux solver 
- **IMEX Integrator Architecture:** Formulated an Implicit-Explicit Runge-Kutta staging structure (specifically the ARS-2,2,2 scheme) using sparse matrix linear solvers for parabolic diffusion terms

### Technical Architecture & Numerical Methods

| Component | Numerical Scheme / Architecture | Engineering Motivation |
| :--- | :--- | :--- |
| **Grid & Memory Layout** | Staggered Mesh (CT), Struct of Arrays (SoA) | Enforces face-centered magnetic flux while maximising CPU cache locality for vectorisation. |
| **Spatial Reconstruction** | 2nd-Order TVD MUSCL (Minmod Limiter) | Prevents non-physical oscillations across steep gradients and shock fronts. |
| **Riemann Flux Solver** | HLL Approximate Riemann Solver | Resolves non-linear intermediate wave states without expensive full-eigensystem solves. |
| **Divergence Constraint** | Upwind Constrained Transport (UCT) | Preserves $\nabla \cdot \mathbf{B} = 0$ to machine precision, preventing non-physical magnetic monopoles. |
| **Time Integration** | ARS(2,2,2) IMEX Runge-Kutta | Decouples non-stiff advective fluxes from stiff diffusion terms to prevent restrictive CFL limits. |
| **Linear System Solver** | Sparse Matrix Solvers (Eigen Library) | Inverts parabolic operators efficiently during stiff implicit stages. |

---

**Core Design Choices**

* **Struct of Arrays (SoA) Layout:** Storing fields in contiguous 1D memory buffers (`rho`, `mx`, `my`, `bx`, `by`, `E`) instead of Arrays of Structures (AoS) ensures sequential memory access during spatial sweeps, enabling CPU auto-vectorisation (SIMD).
* **Shock Capturing & Solenoidal Protection:** Slope-limited MUSCL reconstruction paired with an HLL Riemann solver sharply resolves hydro-magnetic shocks. Corner-centred electric field evaluations ($E_z$) on a staggered grid guarantee that $\nabla \cdot \mathbf{B} = 0$ is preserved down to machine precision ($\sim 10^{-15}$).
* **Stiff Operator Handling (IMEX Framework):** Isolates stiff parabolic magnetic diffusion ($\eta \nabla^2 \mathbf{B}$) into an implicit operator while updating hyperbolic advection terms explicitly via twin Butcher tableaus. The resulting linear system $(I - \gamma \Delta t \eta \mathbf{L})\mathbf{B}^{n+1} = \mathbf{B}^*$ is solved using Eigen’s sparse matrix modules.

## Validation & Physics Benchmarks

### 1. Canonical Shock-Capturing: 2D Orszag-Tang Vortex
The primary validation benchmark for 2D periodic Ideal Magnetohydrodynamics (MHD). Used to verify multi-scale shock-vortex interactions, magnetic reconnection zones, and high-order symmetry preservation across a closed $2\pi \times 2\pi$ domain.

* **Solenoidal Constraint:** Preserves solenoidal invariants down to machine precision ($\| \nabla \cdot \mathbf{B} \|_\infty \approx 10^{-12} \text{ to } 10^{-15}$) across full simulation runs via Staggered-Grid Upwind Constrained Transport (UCT).
* **Spatial Symmetries:** Strictly preserves 180° point-reflection symmetry about the domain center $(\pi, \pi)$ and quadrant parity across $x=y$ and $x=-y$ diagonals throughout quadrupolar shock formation.
* **Shock Capturing:** Resolves sharp supersonic density shocks and current sheets without generating non-physical oscillations or negative pressures.

---

### 2. Macroscopic Plasma Instabilities: Pinch & Buckling Modes
Evaluates the solver's ability to model non-linear topological breakdowns in magnetised plasma columns.

#### (i) Sausage Instability ($m=0$) Compression Benchmark

* **Initial Plasma Beta ($\beta$):** Low-beta regime ($p_0 = 0.1, B_0 = 5.0$) to enable strong magnetic confinement over thermal pressure.
* **Core Compression Ratio:** **3.37×** increase in peak thermal pressure ($p_{\text{neck}}(1.5 $$\mathrm s$$ ) / p_{\text{neck}}(0) = 3.37$) at the narrowest neck constriction.

#### (ii) Kink Instability ($m=1$):
*  Seeded via transverse displacement; models field-line bunching on the inner radius of a bend, driving runaway $S$-shaped helical buckling toward domain boundaries.


---

### 3. Diffusion & Operator Convergence: Decaying Magnetic Sine Wave
Verifies the accuracy and convergence order of the implicit Runge-Kutta staging structure (ARS-2,2,2) when solving parabolic diffusion operators ($\eta \nabla^2 \mathbf{B}$).

* **Analytical Benchmark:** Tests field decay against the exact analytical solution:
  $$B(x, y, t) = B_0 \sin(k_x x) \sin(k_y y) e^{-\eta (k_x^2 + k_y^2)t}$$
* **Convergence Verification:** Confirms second-order temporal convergence ($\mathcal{O}(\Delta t^2)$) by evaluating $L_2$ error norms across successive grid refinement steps.

## Bulid

```bash
git clone https://github.com/RikibabManos/mhd-simulator.git
cd mhd-simulator
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

```