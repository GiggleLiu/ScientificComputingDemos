function trace(f, args...)
    untracked_args = untrack.(args)
    @debug "tracking `$f$((untracked_args...,)))`"
    ret, pullback = rrule(f, untracked_args...)
    tracked = Record(f, args, pullback, zero(ret), false)
    return if ret isa AbstractArray
        TrackedArray(ret, tracked)
    elseif ret isa Real
        TrackedReal(ret, tracked)
    elseif ret isa Complex
        TrackedComplex(ret, tracked)
    else
        error("unsupported return type $(typeof(ret))")
    end
end

accumulate_grad!(x::Number, grad) = x + grad
accumulate_grad!(x::AbstractArray, grad) = (x .+= grad)

backpropagate!(tracked_value::Tuple, _grad) = backpropagate!.(tracked_value, ChainRules.unthunk(_grad))
function backpropagate!(tracked_value, _grad)
    @show tracked_value
    grad = unthunk(_grad)
    is_tracked(tracked_value) || error("expect tracked value")
    record = tracked_value.record::Record
    record.grad = accumulate_grad!(record.grad, grad)
    record.is_leaf && return

    # we won't have callable objects in our tape
    # since we always trace to TrackedType
    @debug "back propagating `pullback$((record.f, untrack.(record.args)...))($grad)`"
    grad_args = Base.tail(record.pullback(record.grad))
    for (arg, grad_arg) in zip(record.args, grad_args)
        is_tracked(arg) && backpropagate!(arg, grad_arg)
    end
    return
end

function gradient(f, args...)
    tracked_args = track.(args)
    ret = f(tracked_args...)
    # let's only support complex-value params not returned loss
    ret isa Real || error("cannot differentiate return type $(typeof(ret)), expect a `Real`")
    backpropagate!(ret, one(untrack(ret)))
    return map(tracked_args) do arg
        arg.record.grad
    end
end
