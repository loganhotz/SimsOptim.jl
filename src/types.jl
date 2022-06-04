abstract type SimsOptimResults end

struct OptimizationResults{Tx, Tf, Tg} <: SimsOptimResults
	method::SimsOptimMethod
	# 
	iterations::Int
	converged::Bool
	#
	value::Real
	minimum::Tf
	#
	minimizer::Tx
	initial_x::Tx
	#
	x_converged::Bool
	x_change::Tx
	#
	f_converged::Bool
	f_change::Tf
	#
	g_converged::Bool
	g_size::Tg
	#
	H::Union{AbstractMatrix, Nothing}
	#
	f_calls::Int
	g_calls::Int
	#
	flag::Int
	message::AbstractString
end



converged(rs::SimsOptimResults) = rs.converged
x_converged(rs::SimsOptimResults) = rs.x_converged
f_converged(rs::SimsOptimResults) = rs.f_converged
g_converged(rs::SimsOptimResults) = rs.g_converged

method(rs::SimsOptimResults) = rs.method
flag(rs::SimsOptimResults) = rs.flag
message(rs::SimsOptimResults) = rs.message

value(rs::SimsOptimResults) = rs.value
minimum(rs::SimsOptimResults) = rs.minimum
minimizer(rs::SimsOptimResults) = rs.minimizer

iterations(rs::SimsOptimResults) = rs.iterations
f_calls(rs::SimsOptimResults) = rs.f_calls
g_calls(rs::SimsOptimResults) = rs.g_calls

initial_x(rs::SimsOptimResults) = rs.initial_x
x_change(rs::SimsOptimResults) = rs.x_change
f_change(rs::SimsOptimResults) = rs.f_change
g_size(rs::SimsOptimResults) = rs.g_size



x_obj(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real} = maximum(abs, x .- y)
x_obj(x::AbstractVector{T}) where {T<:Real} = sum(abs.(x))
f_obj(f::AbstractVector{T}, g::AbstractVector{T}) where {T<:Real} = maximum(abs, f .- g)
f_obj(f::AbstractVector{T}) where {T<:Real} = sum(abs.(f))
f_obj(f::T, g::T) where {T<:Real} = abs(f - g)
f_obj(f::T) where {T<:Real} = abs(f)
g_obj(g::AbstractArray{T}) where {T<:Real} = maximum(abs, vec(g))

function x_converged(x::AbstractVector{T}, y::AbstractVector{T}, x_tol::T) where {T<:Real}
	return x_obj(x, y) < x_tol
end
x_converged(x::AbstractVector{T}, x_tol::T) where {T<:Real} = x_obj(x) < x_tol


function f_converged(f::AbstractVector{T}, g::AbstractVector{T}, f_tol::T) where {T<:Real}
	return f_obj(f, g) < f_tol
end
f_converged(f::AbstractVector{T}, f_tol::T) where {T<:Real} = f_obj(f) < f_tol
f_converged(f::T, g::T, f_tol::T) where {T<:Real} = f_obj(f, g) < f_tol
f_converged(f::T, f_tol::T) where {T<:Real} = f_obj(f) < f_tol


g_converged(g::AbstractArray{T}, g_tol::T) where {T<:Real} = g_obj(g) < g_tol



function Base.show(io::IO, rs::SimsOptimResults)

	converged(rs) ? (status_string = "success") : "failure"
	println("status using $(method(rs)): $status_string")

	println("\nconvergence")
	println("----------------")
	println("    x: $(x_converged(rs))")
	println("    f: $(f_converged(rs))")
	println("    g: $(g_converged(rs))")

	println("\nobjectives")
	println("----------------")
	println("    value    : $(value(rs))")
	println("    minimum  : $(minimum(rs))")
	println("    minimizer: $(minimizer(rs))")

	println("\ncounts")
	println("----------------")
	println("    total iterations: $(iterations(rs))")
	println("    f calls         : $(f_calls(rs))")
	println("    g calls         : $(g_calls(rs))")

	println("\nimprovements")
	println("----------------")
	println("    initial_x: $(initial_x(rs))")
	println("    x_change : $(x_change(rs))")
	println("    f_change : $(f_change(rs))")
	println("    g_size   : $(g_size(rs))")

	println("\ninformation")
	println("----------------")
	println("    flag   : $(flag(rs))")
	println("    message: $(message(rs))")
end
