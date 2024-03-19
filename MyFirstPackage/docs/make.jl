using MyFirstPackage
using Documenter

DocMeta.setdocmeta!(MyFirstPackage, :DocTestSetup, :(using MyFirstPackage); recursive=true)

makedocs(;
    modules=[MyFirstPackage],
    authors="GiggleLiu",
    sitename="MyFirstPackage.jl",
    format=Documenter.HTML(;
        canonical="https://GiggleLiu.github.io/MyFirstPackage.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/GiggleLiu/MyFirstPackage.jl",
    devbranch="main",
)
