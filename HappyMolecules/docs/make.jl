using Documenter, HappyMolecules

makedocs(;
    modules=[HappyMolecules],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Reference" => "ref.md",
    ],
    repo="https://github.com/CodingThrust/HappyMolecules.jl/blob/{commit}{path}#L{line}",
    sitename="HappyMolecules.jl",
    authors="GiggleLiu",
)

deploydocs(;
    repo="github.com/CodingThrust/HappyMolecules.jl",
)
