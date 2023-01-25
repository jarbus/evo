module EvoTrade

# args, net, multiproc
export get_arg_table, parse_args, make_procs, make_model
# Utils
export mk_env_config, ts, log_mmm!, aggregate_metrics, 
aggregate_rollouts, update_df_and_write_metrics,
average_bc, average_walk, max_bc
# maze
export maze_from_file, step!, reset!, sample_batch, 
get_obs, MazeEnv, print_maze
# trade
export reset!, step!, batch_reset!, batch_step!, PyTrade, 
render, get_metrics, batch_pos!
# viz
export plot_grid_and_walks, vis_outs
# construction
export reconstruct, SeedCache, cache_elites!, rm_params,  ModelInfo, gen_params
export compute_novelty, compute_novelties, bc1, bc2, bc3,
create_next_pop, add_to_archive!, M, reorder!,
elite, mr, create_rollout_groups, aid, 
compute_prefixes, decompress_group, add_elite_idxs, compress_pop,
all_v_all, singleton_groups
export run_batch, invert
export EvoTradeLogger

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
include("utils.jl")
include("args.jl")
include("multiproc.jl")
include("logger.jl")
include("net.jl")
include("construction.jl")
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
