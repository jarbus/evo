using Revise
includet("args.jl")
file = read(pipeline(`ls afiles`, `fzf`), String) |> strip
expname = ["--exp-name", file[1:end-4], "--local"] # get rid of .arg
arg_vector = read("afiles/$file", String) |> split
args = parse_args(vcat(arg_vector, expname), s)
includet("evotrade.jl")
