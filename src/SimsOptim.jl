module SimsOptim

using ForwardDiff
using LinearAlgebra
using PrecompileTools
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

@setup_workload begin
    cs = Csminwel()
    x0 = [5.4, 9.2]
    rosen(x) = 100*(x[2] - x[1])^2 + (1 - x[1])^2

    @compile_workload begin
        res = optimize(rosen, x0, cs; iterations = 400)
    end
end


end # module
