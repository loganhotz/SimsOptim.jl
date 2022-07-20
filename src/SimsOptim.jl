module SimsOptim

using ForwardDiff
using LinearAlgebra
using Random

using Formatting: printfmtln

# extending REPL display functions
import Base: show

export optimize
export Csolve
export Csminwel
export OptimizationResults


abstract type SimsOptimMethod end
struct Csolve <: SimsOptimMethod end
struct Csminwel <: SimsOptimMethod end


include("types.jl")
include("algos.jl")


end # module
