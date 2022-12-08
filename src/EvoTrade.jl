module EvoTrade

export get_arg_table, parse_args, make_procs, make_model
export mk_env_config, llog, update_df, write_mets, ts
export maze_from_file, step!, reset!, sample_batch, get_obs, MazeEnv, print_maze
export batch_reset!, batch_step!, PyTrade, render, get_metrics, plot_bcs
export compute_centered_ranks, NoiseTable, compute_grad, get_noise
export reconstruct, compute_novelty, bc1, create_next_pop, add_to_archive!,
reorder!, compute_elite
export run_batch

using ArgParse
using Infiltrator
using Dates
using CSV
using Flux
using Distributed
using DataFrames
using Statistics
using Printf
include("utils.jl")
include("args.jl")
include("multiproc.jl")
include("net.jl")
using .Net

include("trade.jl")
using .Trade
include("maze.jl")
using .Maze
include("es.jl")
include("ga.jl")
using .ES
using .GANS
include("rollout.jl")
include("visual.jl")


end
