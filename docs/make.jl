push!(LOAD_PATH, "../src")

using Documenter
using SimsOptim

makedocs(
    sitename="SimsOptim",
    modules=[SimsOptim]
)
