#!/usr/bin/env julia
file = read(pipeline(`ls outs`, `fzf`), String) |> strip
checkfile = "outs/$file/check.jld2"
if !isfile(checkfile)
    throw("No file $checkfile")
end
argfile = "afiles/$file.arg"
args = map(strip, readlines(argfile)) .|> split
mazefile = nothing
for arg in args
    if arg[1] == "--maze"
        global mazefile = arg[2]
    end
end

lines = map(strip, readlines(mazefile)) .|> 
    x->split(x,"") .|> 
    x->parse(Int, x)
maze_matrix = hcat(hcat(lines...))

using Plots
using JLD2

hm = heatmap(maze_matrix)
check = load(checkfile)
@assert "archive" in keys(check)
archive = check["archive"]
poses = [(pos[2], pos[1]) for (pos, _) in archive]
p = scatter!(hm, poses, color=:blue)
savefig(p, "outs/$file/maze.png")
