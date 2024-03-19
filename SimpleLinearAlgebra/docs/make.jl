using SimpleLinearAlgebra
using Documenter

DocMeta.setdocmeta!(SimpleLinearAlgebra, :DocTestSetup, :(using SimpleLinearAlgebra); recursive=true)

makedocs(;
    modules=[SimpleLinearAlgebra],
    authors="YidaiZhang",
    sitename="SimpleLinearAlgebra.jl",
    format=Documenter.HTML(;
        canonical="https://YidaiZhang.github.io/SimpleLinearAlgebra.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/YidaiZhang/SimpleLinearAlgebra.jl",
    devbranch="main",
)
