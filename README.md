# GoldfarbIdnaniQPSolver.jl

This repository implements the quadratic programming solver for the problem
$$\min_{u \in \mathbb{R}^{n}} \frac{1}{2} u^{T} M u - u^{T} v, \qquad \text{subject to } A^{T} u \geq b$$
proposed by Goldfarb and Idnani, see [A numerically stable dual method for solving strictly convex quadratic programs (1983)](https://doi.org/10.1007/BF02591962).

This is a port of Berwin A. Turlach [quadprog](https://github.com/cran/quadprog) to the Julia language, inspired by the Julia implementation [GoldfarbIdnaniSolver.jl](https://github.com/fabienlefloch/GoldfarbIdnaniSolver.jl). The main difference is the data structures of the program and that this version uses caching to avoid allocations for scenarios that require solving many quadratic problems. This is especially advantageous when the underlying matrix is the same across calls.

Here is a simple example, solving the problem corresponding to $\min \|u - v\|^{2}$ subject to $u_{1} >= 1$ and $u_{2} >= 1$, where $v = [-1, 3, 2]$. The solution is $[1, 3, 2]$.
```julia
n = 3
M = Diagonal(ones(3))
A = spzeros(3, 2)
A[1, 1] = 1
A[2, 2] = 1
b = [1, 1]
cache = initialise(M, A, b)
v = [-1.0, 3.0, 2.0]
v, cache = solve!(v, cache)
```

Another example corresponding to the projection of mesh parameters with minimum mesh size is available [here](examples/meshProjection.jl).