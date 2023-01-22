function get_afiles()
    afiles = []
    for class_dir in readdir("./afiles")
        !isdir("./afiles/$class_dir") && continue
        for arg in readdir("./afiles/$class_dir")
            if occursin(r"arg$", arg)
                push!(afiles, "./afiles/$class_dir/$arg")
            end
        end
    end
    afiles
end
# concat all file names into a string separated by \n
afile_str = join(get_afiles(),"\n")
# pipe that into fzf to choose a file
file = read(pipeline(`echo $afile_str`, `fzf`), String) |> strip
exp_name = basename(file)
cls_name = basename(dirname(file))

using Revise
using Dates
using EvoTrade
expname = ["--exp-name", exp_name, "--cls-name", cls_name, "--local", "--datime", Dates.format(now(), "mm-dd_HH:MM")] # get rid of .arg
arg_vector = read(file, String) |> split
lines = readlines(file) .|> strip
arg_vector = []
for line in lines
    # ignore lines that start with #
    occursin(r"^#", line) && continue
    append!(arg_vector, split(line))
end
println(arg_vector)
args = parse_args(vcat(arg_vector, expname), get_arg_table())
make_procs(args)
includet("x/"*args["algo"]*"trade.jl")
