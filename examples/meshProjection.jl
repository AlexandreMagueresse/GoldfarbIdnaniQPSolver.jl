using QP
using LinearAlgebra
using SparseArrays
using Plots

n = 15
Ω = (-1, +1)
meas = Ω[2] - Ω[1]
hmin = min(1.0e-1, meas / 4 / (n - 1))
zero_trace = false
δ = zero_trace ? 0 : meas / 10

# Automatic
ε = min(1.0e-3, meas / 100 / (n - 1))
Ω₁ = (Ω[1] - δ + ε, Ω[2] + δ - ε)
Ω₂ = zero_trace ? () : (Ω[1], Ω[2])

M = Diagonal(ones(n))
numConstraints = zero_trace ? n + 1 : n + 3
A = spzeros(Float64, n, numConstraints)
b = zeros(Float64, numConstraints)
for i in 1:n-1
  A[i, i] = -1
  A[i+1, i] = 1
  b[i] = hmin
end

A[1, n] = 1
b[n] = Ω₁[1]

A[n, n+1] = -1
b[n+1] = -Ω₁[2]

if !isempty(Ω₂)
  A[2, n+2] = 1
  b[n+2] = Ω₂[1]

  A[n-1, n+3] = -1
  b[n+3] = -Ω₂[2]
end

cache = initialise(M, A, b)

plot()

v = sort!(Ω₁[1] - δ .+ (Ω₁[2] - Ω₁[1] + 2 * δ) .* rand(n))
minBefore, maxBefore = v[1], v[n]
scatter!(v, 0.2 .* ones(n), c=:blue, label="Original")

plot!([Ω₁[1], Ω₁[1]], [-1, 1], ls=:dash, lw=1, c=:black, label="")
plot!([Ω₁[2], Ω₁[2]], [-1, 1], ls=:dash, lw=1, c=:black, label="")
if !zero_trace
  plot!([Ω₂[1], Ω₂[1]], [-1, 1], ls=:dash, lw=1, c=:black, label="")
  plot!([Ω₂[2], Ω₂[2]], [-1, 1], ls=:dash, lw=1, c=:black, label="")
end

v, cache = solve!(v, cache)
scatter!(v, -0.2 .* ones(n), c=:green, label="Projected")

display(plot!(xlims=(min(Ω₁[1] - δ, minBefore), max(maxBefore, Ω₁[2] + δ))))
display(plot!(ylims=(-1, +1), aspectratio=:equal))
display(plot!(aspectratio=:equal))
savefig("plot.png")
