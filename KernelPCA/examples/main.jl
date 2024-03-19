using KernelPCA, Makie, CairoMakie

function showres(res)
    dataset = res.anchors
    x, y = getindex.(dataset, 1), getindex.(dataset, 2)
    @show res.lambda
    kf = kernelf(res, 1)
    X, Y = minimum(x):0.01:maximum(x), minimum(y):0.01:maximum(y)
    @show X, Y
    #levels = -0.1:0.01:0.1
    plt = Plots.contour(X, Y, kf.(KernelPCA.Point.(X', Y)); label="")
    Plots.scatter!(plt, x, y; label="data")
end


# centered K-PCA
Random.seed!(2)
kernel = PolyKernel{2}()
#kernel = LinearKernel()
dataset = KernelPCA.DataSets.curve(100)
res = kpca(kernel, dataset; centered=true)

Φ = [ϕ(kernel, x) for x in dataset]
Φ = Φ .- Ref(sum(Φ) ./ length(Φ))
C = sum(x->1/length(dataset) .* x * x', Φ)
V1 = sum([alpha * x  for (alpha, x) in zip(res.vectors[:, 1], Φ)])

@info res.lambda
for k in 1:length(res.lambda)
    @test res.lambda[1] * V1 ≈ C * V1
end
display(showres(res))