


csminwel_messages = Dict{Int, String}(
    0 => "success",
    1 => "zero gradient",
    2 => "back and forth on step length never finished",
    3 => "smallest step still improving too slow",
    4 => "back and forth on step length never finished",
    5 => "largest step still improving too fast",
    6 => "smallest step still improving too slow; reversed gradient",
    7 => "warning: possible inaccuracy in H matrix"
)





function optimize(
    f::Function,
    g::Function,
    H0::AbstractMatrix{T},
    x0::AbstractVector{R},
    m::Csminwel;
    f_tol::Real         = 1e-14,
    g_tol::Real         = 1e-8,
    x_tol::Real         = 1e-32,
    iterations::Integer = 100,
    δ::Real             = 1e-6,
    α::Real             = 1e-3,
    verbose::Bool       = false
) where {T, R <: Real}

    # unify types (Float64 type is included in cases, e.g. H0 = I and x0 = ones(nx)
    TR = promote_type(T, R, Float64)
    H0 = convert(Matrix{TR}, H0)
    x0 = convert(Vector{TR}, x0)

    # initial guess for optimization, initial gradient, and flag
    f0   = f(x0)
    g0   = g(x0)
    badg = bad_grad(g0)

    # counting function calls
    f_calls, g_calls = 1, 1

    # loop-managing variables & return flag
    iter     = 0
    not_done = true
    flag     = -1

    fx    = copy(f0)
    gx    = copy(g0)
    x     = copy(x0)
    H     = copy(H0)

    while not_done
        iter += 1

        verbose && println("f at beginning of new iteration = $fx")
        verbose && println("x = ")
        verbose && display(x)
        fx1, x1, fc, flag1 = csmininit(f, x, fx, gx, badg, H, verbose=verbose)

        # keep track of all those function calls
        f_calls += fc

        if flag1 != 1
            if (flag1 == 2) || (flag1 == 4)
                wall1 = true
                badg1 = true

            else
                # approximate the gradient
                gx1     = g(x1)
                g_calls += 1

                badg1 = bad_grad(gx1)
                wall1 = copy(badg1)
    
            end # if flag1 == 2 || flag1 == 4

            if wall1 && (length(H) > 1)
                # bad gradient or back and forth on step length. possibly at a cliff edge
                #    try perturbing search direction if dimensions > 1
                Hcliff = H + Diagonal(H).*rand(nx)
                fx2, x2, fc, flag2 = csmininit(f, x, fx, gx, badg, Hcliff, verbose=verbose)
                verbose && println("cliff. perturbing search directions")

                f_calls += fc
                if fx2 < fx
                    if (flag2 == 2) || (flag2 == 4)
                        wall2 = true
                        badg2 = true

                    else
                        # approximating the gradient
                        gx2     = g(x2)
                        g_calls += 1

                        badg2 = bad_grad(gx2)
                        wall2 = copy(badg2)

                    end # if flag1 == 2 || flag1 == 4

                    if wall2
                        verbose && println("another cliff. traversing...")

                        if norm(x2 - x1) < 1e-13
                            fx3, x3      = fx, x
                            badg3, flag3 = true, -1

                        else
                            gcliff = ((fx2-fx1) / (norm(x2-x1)^2)) .* (x2 - x1)
                            eye = Matrix{eltype(x)}(I, nx, nx)
                            
                            fx3, x3, fc, flag3 = csmininit(
                                f, x, fx, gcliff, false, eye, verbose=verbose
                            )

                            f_calls += fc
                            if (flag3 == 2) || (flag3 == 4)
                                wall3, badg3 = true, true

                            else
                                # approximating the gradient
                                gx3     = g(x3)
                                g_calls += 1

                                badg3 = bad_grad(gx3)
                                wall3 = copy(badg3)

                            end
                        end

                    else
                        fx3, x3      = fx, x
                        badg3, flag3 = true, -1

                    end

                else
                    fx3, x3      = fx, x
                    badg3, flag3 = true, -1

                end # if fx2 < fx

            else
                # normal iteration, no walls, or its 1-dim
                fx2, fx3     = fx, fx
                badg2, badg3 = true, true
                flag2, flag3 = -1, -1

            end # if wall1 && (length(H) > 1)

        else
            fx1, fx2, fx3 = fx, fx, fx
            flag2, flag3  = flag1, flag1

        end # if flag1 != 1

        # now we pick gh and xh
        if (fx3 < (fx - f_tol)) && !badg3
            ih, fxh, gxh, xh = 3, fx3, gx3, x3
            badgh, flagh     = badg3, flag3
    
        elseif (fx2 < (fx - f_tol)) && !badg2
            ih, fxh, gxh, xh = 2, fx2, gx2, x2
            badgh, flagh     = badg2, flag2
    
        elseif (fx1 < (fx - f_tol)) && !badg1
            ih, fxh, gxh, xh = 1, fx1, gx1, x1
            badgh, flagh     = badg1, flag1
    
        else
            # none of our function evals meet criteria
            fxh, ih = findmin((fx1, fx2, fx3))
            verbose && println("ih = $ih")

            if ih == 1
                xh, flagh = x1, flag1

            elseif ih == 2
                xh, flagh = x2, flag2

            else
                xh, flagh = x3, flag3

            end

            gxh     = g(xh)
            g_calls += 1
            badgh   = true
    
        end

        stuck = abs(fxh - fx) < f_tol
        if !badg & !badgh & !stuck
            H = bfgsi(H, gxh - gx, xh - x; verbose=verbose)
        end

        verbose && println("improvement on iteration $iter: $(fx-fxh)")
        iter > iterations ? (not_done = false) : ( stuck ? (not_done = false) : nothing )

        fx   = fxh
        x    = xh
        gx   = gxh
        badg = badgh
        flag = flagh

    end # while not_done

    conv, x_conv, f_conv, g_conv = converged(
        x, fx, gx,
        x_tol, f_tol, g_tol
    )

    return OptimizationResults(
        m, iter, conv,
        fx, fx,
        x, x0,
        x_conv, x0 .- x,
        f_conv, f0 .- fx,
        g_conv, g_obj(gx), H,
        f_calls, g_calls, flag, csminwel_messages[flag]
    )
end



function optimize(
    f::Function,
    g::Function,
    H0::UniformScaling,
    x0::AbstractVector,
    m::Csminwel;
    kwargs...
)

    # just transmute uniform scaling into a matrix
    nx = length(x0)
    H0 = Matrix(H0, nx, nx)

    return optimize(f, g, H0, x0, m; kwargs...)
end



function optimize(
    f::Function,
    g::Function,
    x0::AbstractVector,
    m::Csminwel;
    kwargs...
)
    curvature = 1e-5
    return optimize(f, g, curvature*I, x0, m; kwargs...)
end




function optimize(
    f::Function,
    H0::AbstractMatrix,
    x0::AbstractVector,
    m::Csminwel;
    kwargs...
)
    # default to backend differentiation
    g = derivative(f, m)

    return optimize(f, g, H0, x0, m; kwargs...)
end



function optimize(
    f::Function,
    H0::UniformScaling,
    x0::AbstractVector,
    m::Csminwel;
    kwargs...
)

    # just transmute uniform scaling into a matrix
    nx = length(x0)
    H0 = Matrix(H0, nx, nx)

    return optimize(f, H0, x0, m; kwargs...)
end



function optimize(f::Function, x0::AbstractVector, m::Csminwel; kwargs...)
    curvature = 1e-5
    return optimize(f, curvature*I, x0, m; kwargs...)
end





"""
initialization process for the `csminwel` algorithm that is called in each iteration, and
additionally each time a 'wall' is encounted during the search
"""
function csmininit(
    f::Function,
    x0::Vector{T},
    f0::T,
    g0::Vector{T},
    badg::Bool,
    H0::AbstractMatrix{T};
    verbose::Bool = false
) where {T<:Real}

    angle     = 0.005
    θ         = 0.3
    f_change  = 1_000

    minimum_λ  = 1e-9
    minimum_δf = 0.01

    f_calls = 0
    λ       = 1

    xhat  = copy(x0)
    fx    = copy(f0)
    fhat  = copy(f0)
    g     = copy(g0)
    gnorm = norm(g0)

    if (gnorm < 1e-12) && !badg
        # gradient's converged
        retcode = 1
        dxnorm  = 0

    else
        # two branches from this point: (1) match rate of improvement along directional
        #    derivative if we have a good gradient, or (2) any improvement in f otherwise
        dx     = -(H0 * g)
        dxnorm = norm(dx)
        if dxnorm > 1e12
            verbose && println("near-singular H problem")
            dx = (f_change / dxnorm) .* dx

        end

        dfhat = dot(dx, g0)
        if !badg
            # test for alignment of dx with directional derivative, and fix if necessary
            a = -dfhat / (gnorm*dxnorm)
            if a < angle
                dx = dx .- (angle*dxnorm/gnorm + dfhat/(gnorm*gnorm)) .* g
                dx = (dxnorm/norm(dx)) .* dx # keep scale invariant to the angle correction
                dfhat = dot(dx, g)

                verbose && println("correction for low angle: $a")

            end
        end
        verbose && println("predicted improvement: $(-dfhat/2)")

        # ought to have an okay dx; now we adjust step length λ until min and max
        # improvement rate criteria are met
        done = false
        factor = 3
        shrink = true

        λ_min  = 0
        λ_max  = Inf
        λ_peak = 0

        f_peak = f0
        λ_hat  = 0

        while !done
            dxtest  = x0 + λ*dx
            fx      = f(dxtest)
            f_calls += 1

            verbose && println("λ = $λ, f = $f")

            if fx < fhat
                fhat  = fx
                xhat  = dxtest
                λ_hat = λ

            end

            deriv         = max(-θ*dfhat*λ, 0)
            shrink_signal = ( !badg && ((f0-fx) < deriv) ) || (badg && ((f0-fx)<0))
            grow_signal   = !badg && (λ > 0) && ( (f0-fx) > -(1-θ)*dfhat*λ )

            if shrink_signal && ( (λ > λ_peak) || (λ < 0) )
                if (λ > 0) && ( !shrink || (λ/factor <= λ_peak) )
                    shrink = true
                    factor = factor ^ 0.6
                    while λ/factor <= λ_peak
                        factor = factor ^ 0.6
                    end

                    if abs(factor - 1) < minimum_δf
                        abs(λ) < 4 ? (retcode = 2) : (retcode = 7)
                        done = true

                    end # abs(factor - 1) < minimum_δf
                end # (λ > 0) && (!shrink && (λ/factor <= λ_peak))

                if (λ < λ_max) && (λ > λ_peak)
                    λ_max = λ

                end

                λ = λ / factor
                if abs(λ) < minimum_λ
                    if (λ > 0) && (f0 <= fhat)
                        # try going against the gradient, which may be inaccurate
                        λ = -λ * factor ^ 6

                    else
                        λ < 0 ? (retcode = 6) : (retcode = 3)
                        done = true

                    end # if (λ > 0) && (f0 <= fhat)
                end # if abs(λ) < minimum_λ

            elseif (grow_signal && λ > 0) || (shrink_signal && (λ <= λ_peak) && (λ > 0))
                if shrink
                    shrink = false
                    factor = factor ^ 0.6
                    if abs(factor - 1) < minimum_δf
                        abs(λ) < 4 ? (retcode = 4) : (retcode = 7)
                        done = true

                    end # if abs(factor - 1) < minimum_δf
                end # if shrink

                if (fx < f_peak) && (λ > 0)
                    f_peak = fx
                    λ_peak = λ
                    if λ_max <= λ_peak
                        λ_max = λ_peak * factor^2
                    end
                end

                λ = λ * factor
                if abs(λ) > 1e20
                    retcode = 5
                    done    = true
                end
    
            else
                factor < 1.2 ? (retcode = 7) : (retcode = 0)
                done = true

            end # shrink and grow signals
        end # done
    end # gnorm < 1e-12 && !badg

    verbose && println("norm of dx $dxnorm")

    return fhat, xhat, f_calls, retcode
end





"""
Broyden-Fletcher-Goldfarb-Shanno algorithm update for Hessian matrix
"""
function bfgsi(
    B::AbstractMatrix,
    y::AbstractVector,
    s::AbstractVector;
    verbose::Bool = true
)
    B_y = B * y
    y_s = dot(y, s)

    if abs(y_s) > 1e-12
        cov_term  = (1 + dot(B_y, y) / y_s) .* (s .* s')
        corr_term = B_y .* s' + s .* B_y'
        B_new     = B .+ (cov_term .- corr_term) ./ y_s
    else
        verbose && println("bfgs update failed")
        B_new = B
    end

    return B_new
end
