module AutodiffAndOptimization

export rosenbrock
export simplex, simplex1d

function rosenbrock(x)
    (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

include("simplex.jl")

end
