module QP

using LinearAlgebra
using SparseArrays

include("linalg.jl")
include("solve.jl")

export initialise
export updateM!
export updateMDiag!
export updateA!
export solve!

end # module QP
