function trace(f, args...)
    untracked_args = untrack.(args)
    @debug "tracking `$f$((untracked_args...,)))`"
    ret, pullback = rrule(f, untracked_args...)
    function pullback_unthunk(grad)
        grad_args = pullback(grad)
        return unthunk.(grad_args)
    end
    @assert ret isa AbstractArray "expect the output a primitive function to be an array, got $(typeof(ret))"
    record = Record(f, args, pullback_unthunk, zero(ret), false)
    return TrackedArray(ret, record)
end

backpropagate!(tracked_value::Tuple, grad::AbstractArray) = backpropagate!.(tracked_value, grad)
function backpropagate!(tracked_value, grad::AbstractArray)
    @debug "backpropagate! `$(tracked_value)`"
    is_tracked(tracked_value) || error("expect tracked value")
    record = tracked_value.record
    record.grad .+= grad   # accumulate grad
    record.is_leaf && return
    @debug "back propagating `pullback$((record.f, untrack.(record.args)...))($grad)`"
    grad_args = Base.tail(record.pullback(record.grad))
    for (arg, grad_arg) in zip(record.args, grad_args)
        is_tracked(arg) && backpropagate!(arg, grad_arg)
    end
    return
end

function gradient(f, args::Tuple)
    tracked_args = track.(args)
    ret = f(tracked_args...)
    @assert eltype(ret) <: Real && ndims(ret) == 0 "expect a scalar real output as the loss function! got $(typeof(ret))"
    backpropagate!(ret, fill!(similar(ret), 1))  # the gradient of the loss function is 1
    return map(arg -> arg.record.grad, tracked_args)
end
