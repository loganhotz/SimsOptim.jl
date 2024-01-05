"""
through some trial and error, I've learned that Chris Sims' algorithms seem to work better
using finite differencing methods than with automatic differentiation
"""

const FDM_DEFAULTS = Dict{Symbol, Any}(:p => 5,
                                       :q => 1)



"""
    set_fdm_settings!(k::Symbol, Any)

Update default parameters for the `FiniteDifferences.central_fdm` method. Default settings
are

- `k = 5` the number of grid points used in the approximation algorithm

- `p = 1` the order of the derivative to take. (Not sure why this should be changed)
"""
function set_fdm_settings!(k::Symbol, v::Any)
    haskey(FDM_DEFAULTS, k) || error("unrecognized `fdm` setting: $k")

    FDM_DEFAULTS[k] = v
    return FDM_DEFAULTS
end
function set_fdm_settings!(p::Pair{Symbol, <:Any})
    k, v = p
    set_fdm_settings!(k, v)
end
function set_fdm_settings!(p::NTuple{N, Pair}) where {N}
    for p_ ∈ p
        set_fdm_settings!(p_)
    end
    return FDM_DEFAULTS
end
set_fdm_settings!(p::Pair, args...) = set_fdm_settings!((p, args...))



"""
    derivative(f::Function, m::Csminwel)
    derivative(f::Function, m::Csolve)

Compute the derivative/gradient function of `f`. This uses the `FiniteDifferences.jl`
library functions `central_fdm` and `grad` to approximate the derivative of `f`. The
default parameters of `central_fdm` are `p = 5` and `q = 1`. To update those, see
`set_fdm_settings!`
"""
function derivative(f::Function, m::Csminwel)
    method = central_fdm(FDM_DEFAULTS[:p], FDM_DEFAULTS[:q])
    g(x)   = grad(method, f, x)[1] # not sure why this returns a 1-element tuple?
    
    return g
end
function derivative(f::Function, m::Csolve)
    method = central_fdm(FDM_DEFAULTS[:p], FDM_DEFAULTS[:q])
    g(x)   = jacobian(method, f, x)[1] # again - not sure why this returns a tuple

    # display(g)

    return g
end
