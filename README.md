# SimsOptim.jl

a simple mirror of Chris Sims's `csolve` and `csminwel` optimization functions, originally written in MATLAB, which are available [here](http://sims.princeton.edu/yftp/optimize/).

the interfaces to the `optimize` function and `OptimizationResults` type are based on the analogous objects in the widely-known [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) package, although `SimsOptim.jl` does not import & re-export `Optim.jl`. future versions of `SimsOptim.jl` might do so, however.

the method signatures of `optimize` are
```julia
optimize(
    f::Function,
    g::Function, [optional]
    x0::AbstractVector{T},
    m::Csolve;
    kwargs...
) where {T}
```
for the multidimensional `Csolve` procedure, and
```julia
optimize(
    f::Function,
    g::Function, [optional]
    x0::AbstractVector{T},
    H0::Union{AbstractMatrix{T}, UniformScalint}, [optional]
    m::Csminwel;
    <keyword arguments>
)
```
for the unidimensional `Csminwel` procedure. the keyword arguments are shared between the
two, with default values
```julia
f_tol::Real = 1e-14
g_tol::Real = 1e-8
x_tol::Real = 1e-32
iterations::Int = 1000 (Csolve algorithm) or 100 (Csminwel algorithm)
δ::Real = 1e-6
α::Real = 1e-3
verbose::Bool = false
```

the `ForwardDiff` package's `jacobian` and `gradient` functions are used to approximate `g`
for the `Csolve` and `Csminwel` algorithms, respectively.



# installation
the `SimsOptim` package is registered; simply call `Pkg.add("SimsOptim")` at the REPL



# example
using the two-dimensional Rosenbrock function
```julia
function rosen(z::AbstractVector{T}) where {T}
    a, b = 1, 100
    c, d = a, 150

    x, y = z[1], z[2]

    r = zeros(T, 2)
    r[1] = (a - x^2) + b * (x^2 - y)^2
    r[2] = (c - x^2) + d * (x^2 - y)^2

    return r
end
```
as an example, we can find its minimum easily:
```julia
julia> using Pkg; Pkg.add("SimsOptim")
julia> results = optimize(rosen, [0.5, 4], Csolve())
```
the `optimize` function returns an `OptimizationResults` instance, whose REPL
representation is
```
status using Csolve(): success

convergence
----------------
    x: false
    f: true
    g: false

objectives
----------------
    value    : 5.069629645745414e-15
    minimum  : [2.0278518582981656e-15, 3.0417777874472484e-15]
    minimizer: [1.0, 1.0000000045031676]

counts
----------------
    total iterations: 29
    f calls         : 30
    g calls         : 30

improvements
----------------
    initial_x: [0.5, 4.0]
    x_change : [-0.5, 2.9999999954968324]
    f_change : [1407.0, 2110.125]
    g_size   : 2.00000540380114

information
----------------
    flag   : 0
    message: success
```
each of the listed fields above can be accessed by a non-exported function of the same
name. for example,
```julia
julia> SimsOptim.minimizer(results)
2-element Vector{Float64}:
 1.0
 1.0000000045031676
```
