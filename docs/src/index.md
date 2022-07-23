# SimsOptim.jl



a package for minimizing functions. this is just a translation of Chris Sims' [optimization
code](http://sims.princeton.edu/yftp/optimize/mfiles/), which is widely used in economic
modeling.


## procedures

Professor Sims' `csolve` and `csminwel` algorithms are bundled together into the `optimize`
function of the package, with `m::SimsOptimMethod` argument determining which approach is
used:
```@docs
SimsOptim.optimize
```


## other optimization procedures

the package interface was intentionally designed to mimic that of the more extensive
[Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) package, so `SimsOptim` would
be easier to acclimate to. however, since this package is much more narrow in scope,
`Optim` is not a dependency of this package.


## example

we minimize the Rosenbrock function
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
since this function is vector-valued, we use the `Csolve` method to minimize it:
```julia-shell
julia> using Pkg; Pkg.add("SimsOptim")
julia> optimize(rosen, [3.4, 2.2], Csolve())
```
which returns an `OptimizationResults` object, whose REPL representation is
```
status using Csolve(): success

convergence
----------------
    x: false
    f: true
    g: false

objectives
----------------
    value    : 3.533332217981906e-15
    minimum  : [1.4133328871927622e-15, 2.1199993307891436e-15]
    minimizer: [1.0, 0.999999996240568]

counts
----------------
    total iterations: 32
    f calls         : 33
    g calls         : 33

improvements
----------------
    initial_x: [3.4, 2.2]
    x_change : [2.4, 1.200000003759432]
    f_change : [8750.4, 13130.88]
    g_size   : 1.9999969924544097

information
----------------
    flag   : 0
    message: success
```


## installation

the `SimsOptim` package is registered, simply call `using Pkg; Pkg.add("SimsOptim")` at the
REPL to install
