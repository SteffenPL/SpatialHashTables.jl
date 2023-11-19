using SpatialHashTables
using Documenter

DocMeta.setdocmeta!(SpatialHashTables, :DocTestSetup, :(using SpatialHashTables); recursive=true)

makedocs(;
    modules=[SpatialHashTables],
    authors="Steffen Plunder <steffen.plunder@web.de> and contributors",
    repo="https://github.com/SteffenPL/SpatialHashTables.jl/blob/{commit}{path}#{line}",
    sitename="SpatialHashTables.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SteffenPL.github.io/SpatialHashTables.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/SteffenPL/SpatialHashTables.jl",
    devbranch="main",
)
