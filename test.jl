using Revise
using ProfileView
using BenchmarkTools
using Evo
using Flux
using StableRNGs
using Test
using Flux

ENV["JULIA_STACKTRACE_ABBREVIATED"] = true
ENV["JULIA_STACKTRACE_MINIMAL"] = true
using AbbreviatedStackTraces

# file = "test/test_noisetable.jl"
# file = "test/test_net.jl"
# file = "test/test_maze.jl"
# file = "test/test_trade.jl"
# file = "test/test_rollout.jl"
# file = "test/test_ga.jl"
function t()
    file = joinpath("test/",read(pipeline(`ls test`, `fzf`), String)) |> strip
    println(file)
    include(file)
    roc([file], [Evo]) do
        println("Running main()")
        include(file)
    end
end
