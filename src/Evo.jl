module Evo

export get_arg_table, parse_args, make_procs, make_model
export mk_env_config, update_df, write_mets, ts, log_mmm!
export maze_from_file, step!, reset!, sample_batch, get_obs, MazeEnv, print_maze
export reset!, step!, batch_reset!, batch_step!, PyTrade, render, get_metrics, batch_pos!
export plot_grid_and_walks, vis_outs
export compute_centered_ranks
export reconstruct, SeedCache, cache_elites!
export ModelInfo, gen_params
export compute_novelty, compute_novelties, bc1, bc2, bc3,
create_next_pop, add_to_archive!, M, reorder!, compute_elite,
elite, mr, create_rollout_groups, average_bc, aid, average_walk, 
max_bc, add_elite_idxs_to_groups, compute_prefixes,
compress_groups, decompress_group, add_elite_idxs, compress_pop
export run_batch, invert
export EvoLogger
export make, step!, reset!

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
using LoggingExtras

function reset! end
function step! end

include("utils.jl")
include("args.jl")
include("multiproc.jl")
include("logger.jl")
include("net.jl")
include("construction.jl")
using .Net



include("gym.jl")
include("trade.jl")
include("maze.jl")
include("es.jl")
include("ga.jl")
using .Gym
using .Trade
using .Maze
using .ES
using .GANS
include("rollout.jl")
include("visual.jl")
end
