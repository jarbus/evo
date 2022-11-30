file = read(pipeline(`ls afiles`, `fzf`), String) |> strip
using Revise
using Dates
includet("args.jl")
expname = ["--exp-name", file[1:end-4], "--local", "--datime", Dates.format(now(), "mm-dd_HH:MM")] # get rid of .arg
arg_vector = read("afiles/$file", String) |> split
args = parse_args(vcat(arg_vector, expname), s)
includet(args["algo"]*"trade.jl")
