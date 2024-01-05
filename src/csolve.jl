


csolve_messages = Dict{Int, String}(
    0 => "success",
    1 => "'factor' unchanged",
    2 => "'factor' changed",
    3 => "negative λ",
    4 => "'iterations' exceeded"
)



"""
    optimize(f::Function, x0::Vector, ::Csolve; kwargs...)
    optimize(f::Function, g::Function, x0::Vector, ::Csolve; kwargs...)

---

    optimize(f::Function, x0::Vector, ::Csminwel; kwargs...)
    optimize(f::Function, H0::Matrix,  x0::Vector, ::Csminwel; kwargs...)
    optimize(f::Function, g::Function, x0::Vector, ::Csminwel; kwargs...)
    optimize(f::Function, g::Function, H0::Matrix, x0::Vector, ::Csminwel; kwargs..)

solves the system `f(x) = 0` using Chris Sims' `csolve` method or the scalar equation
`f(x) = 0` using his `csminwel` procedure

# Arguments
- `f::Function`: the objective function to minimize
- `g::Function`: a function that computes the gradient of `f`. if not provided, the
    `jacobian` or `gradient` method from the `FiniteDifferences` package is used to
    approximate `g`, as appropriate
- `x0::AbstractVector`: an initial guess for the optimizing input
- `H0::AbstractMatrix`: an initial guess for the Hessian matrix. the Hessian is only used
    in the `Csminwel` procedure, and if it is not provided, the default is `1e-5 * I`
- `m::SimsOptimMethod`: an instance of the `Csolve` or `Csminwel`  structure

# Keyword Arguments
- `f_tol::Real = 1e-14`: tolerance for successive evaluations of the objective function
- `g_tol::Real = 1e-8`: tolerance for norm of objective's gradient
- `x_tol::Real = 1e-32`: tolerance for successive steps of the input
- `iterations::Int`: maximum number of steps to take. when `isa(m, Csolve)`, the default
    value is 1000, and when `isa(m, Csminwel)`, the default is 100
- `δ::Real = 1e-6`: differencing interval for the numerical gradient
- `α::Real = 1e-3`: tolerance on the rate of descent
- `verbose::Bool = false`: print messages to REPL
"""
function optimize(
    f::Function,
    g::Function,
    x0::AbstractVector{T},
    m::Csolve;
    f_tol::Real         = 1e-14,
    g_tol::Real         = 1e-8,
    x_tol::Real         = 1e-32,
    iterations::Integer = 1_000,
    δ::Real             = 1e-6,
    α::Real             = 1e-3,
    verbose::Bool       = false
) where {T<:Real}

    x  = copy(x0)
    nx = length(x0)

    # initial optimization & gradient, and f-value that will be successively overwritten
    f0 = f(x0)
    fx = copy(f0)
    gx = g(x0)

    # function call counters for objective & gradient
    f_calls, g_calls = 1, 1

    # objective values at consecutive iterations & return flag
    obj, obj_prev = Inf, Inf
    flag          = -1

    # loop-managing variables & printing flags
    conv         = false
    iter         = 0
    imag_verbose = 0
    rand_verbose = 0

    if verbose
        iter_head = rpad("iters", max(trunc(Int, log10(iterations) + 1), 12 + 2))
        ftol_head = rpad("f_tol", 12 + 2)
        flex_head = rpad("λ", 7 + 2)

        header = iter_head * ftol_head * flex_head
        line   = repeat('-', length(header))
        println(header)
        println(line)

        fmt = "{1:>12d}  {2:12.6f}  {3:7.4f}"

    end

    # flags that need to be available after the while loop
    x_conv, f_conv, g_conv = false, false, false

    while !conv
        iter += 1
        if iter > 3 && (obj_prev-obj) < f_tol*max(1, obj) && rem(iter, 2) == 1
            randomize = true

        else
            # approximate the jacobian of the matrix
            gx      = g(x)
            g_calls += 1

            # sanity checks on the jacobian
            if eltype(gx) <: Real
                if 1/cond(gx) < 1e-12
                    gx = gx + UniformScaling(δ)

                end
                dx        = - gx \ fx
                randomize = false

            else
                imag_verbose = iter
                randomize    = true

            end
        end

        if randomize
            rand_verbose = iter
            dx           = norm(x) ./ randn(nx)

        end

        λ, λ_min              = 1, 1
        x_min, f_min, obj_min = x, fx, obj

        step_size       = norm(dx)
        φ               = 0.6
        shrink          = true
        subprocess_done = false

        while !subprocess_done
            λx      = x + λ.*dx
            f_λx    = f(λx)
            obj_λx  = f_obj(f_λx)
            f_calls += 1

            if obj_λx < obj_min
                λ_min   = λ
                x_min   = λx
                f_min   = f_λx
                obj_min = obj_λx

            end

            if ((λ > 0) && ((obj-obj_λx) < α*λ*obj)) || ((λ < 0) && (obj-obj_λx < 0))
                if !shrink
                    φ = φ ^ 0.6
                    shrink = true

                end

                if abs(λ*(1-φ)*step_size) > δ / 10
                    λ = φ * λ

                elseif (λ > 0) && (φ == 0.6)
                    # have only been shrinking at this point
                    λ = -0.3

                else
                    subprocess_done = true
                    λ > 0 ? ( φ == 0.6 ? (flag = 2) : (flag = 1) ) : flag = 3

                end

            elseif (λ > 0) && ((obj_λx-obj) > (1-α)*λ*obj)
                if shrink
                    φ      = φ ^ 0.6
                    shrink = false

                end
                λ = λ / φ

            else
                subprocess_done = true
                flag            = 0

            end # scaling conditions
        end # while !subprocess_done

        conv, x_conv, f_conv, g_conv = converged(
            x - x_min, f_min, gx,
            x_tol, f_tol, g_tol
        )

        # record the improvements for next iteration
        x        = x_min
        fx       = f_min
        obj_prev = obj
        obj      = obj_min

        if verbose
            if rand_verbose == iter
                printfmtln(fmt * " (randomized)", iter, obj, λ)
            elseif imag_verbose == iter
                printfmtln(fmt * " (randomized) (imag. jacobian)", iter, obj, λ)
            else
                printfmtln(fmt, iter, obj, λ)
            end

        end

        if iter >= iterations
            conv, flag = true, 4

        elseif f_conv
            conv, flag = true, 0

        end

    end # while not_done

    return OptimizationResults(
        m, iter, conv,
        obj, fx,
        x, x0,
        x_conv, x0 .- x,
        f_conv, f0 .- fx,
        g_conv, g_obj(gx), nothing,
        f_calls, g_calls, flag, csolve_messages[flag]
    )
end




function optimize(
    f::Function,
    x0::AbstractVector{T},
    m::Csolve;
    kwargs...
) where {T<:Real}
    
    # default to backend differentiation
    g = derivative(f, m)

    return optimize(f, g, x0, m; kwargs...)

end
