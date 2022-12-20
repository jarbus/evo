module EvoTrade

export get_arg_table, parse_args, make_procs, make_model
export mk_env_config, llog, update_df, write_mets, ts
export maze_from_file, step!, reset!, sample_batch, get_obs, MazeEnv, print_maze
export reset!, step!, batch_reset!, batch_step!, PyTrade, render, get_metrics, batch_pos!
export plot_bcs, plot_walks
export compute_centered_ranks
export NoiseTable, compute_grad, get_noise, reconstruct, SeedCache, cache_elites!
export ModelInfo, gen_params
export compute_novelty, compute_novelties, bc1, create_next_pop, add_to_archive!, M, reorder!, compute_elite
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
using Plots
include("utils.jl")
include("args.jl")
include("multiproc.jl")
include("net.jl")
include("noisetable.jl")
using .Net
using .NoiseTables


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
