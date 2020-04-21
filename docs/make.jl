push!(LOAD_PATH,joinpath(@__DIR__, ".."))
using Documenter, Nystrom

makedocs(
    modules = [Nystrom],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Luiz M. Faria and Carlos Perez-Arancibia",
    sitename = "Nystrom.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/maltezfaria/Nystrom.jl.git",
)
