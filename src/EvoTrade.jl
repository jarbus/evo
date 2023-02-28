module EvoTrade

# args, net, multiproc
export get_arg_table, parse_args, make_procs, make_model
# Utils
export mk_env_config, ts, log_mmm!, aggregate_metrics, 
aggregate_rollouts!, update_df_and_write_metrics,
average_bc, average_walk, max_bc, v32, mk_id_player_map,
mmms
# maze
export maze_from_file, step!, reset!, sample_batch, 
get_obs, MazeEnv, print_maze
# trade
export reset!, step!, batch_reset!, batch_step!, PyTrade, 
render, get_metrics, batch_pos!
# viz
export plot_grid_and_walks, vis_outs, plot_bcs
# construction
export reconstruct!, SeedCache, cache_elites!, rm_params,
ModelInfo, gen_params, construct!
# ga
export compute_novelties!, compute_fitnesses!, walks,
create_next_pop, add_to_archive!, M, elite, fitnesses,
mr, create_rollout_groups, aid, update_pops!, novelties,
all_v_all, singleton_groups, random_groups, all_v_best,
bcs

# compression
export compute_prefixes, decompress_group, add_elite_idxs,
compress_pop, compress_pops, compress_elites
# rollout & logger
export run_batch, mk_mods, process_batch
export EvoTradeLogger
# noisetable
export NoiseTable
# genepool
export log_improvements

# structs
export Ind, Pop, BC, F, Geno, CompGeno, RolloutInd,
Prefixes, V32, Batch, compute_compression_data, Mut,
Seed, MR, MutCore, MutBinding, EliteIdxs


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
using Random
using Optionals
using LRUCache
include("structs.jl")
include("utils.jl")
include("genepool.jl")
include("noisetable.jl")
include("args.jl")
include("multiproc.jl")
include("logger.jl")
include("net.jl")
include("construction.jl")
include("trade.jl")
using .Trade
include("maze.jl")
using .Maze
include("ga.jl")
# include("compression.jl")
include("compression3.jl")
include("rollout.jl")
include("visual.jl")
end
