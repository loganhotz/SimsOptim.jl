# SimsOptim.jl

a simple mirror of Chris Sims's `csolve` and `csminwel` optimization functions, originally written in MATLAB, which are available [here](http://sims.princeton.edu/yftp/optimize/).

the interfaces to the `optimize` function and `OptimizationResults` type are based on the analogous objects in the widely-known [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) package, although `SimsOptim.jl` does not import & re-export `Optim.jl`. future versions of `SimsOptim.jl` might do so.

the method signatures of `optimize` are
```julia
optimize(
    f::Function,
    g::Function, [optional]
    x::Vector,
    m::Csolve;
    <keyword arguments>
)
```
and
```julia
optimize(
    f::Function,
    g::Function, [optional]
    x::Vector,
    H::Matrix, [optional]
    m::Csolve;
    <keyword arguments>
)
```
the keyword arguments and their default values are
```julia
f_tol::Real = 1e-14
g_tol::Real = 1e-8
x_tol::Real = 1e-32
iterations::Int = 1000 (Csolve algorithm) or 100 (Csminwel algorithm)
δ::Real = 1e-6
α::Real = 1e-3
verbose::Bool = false
```
