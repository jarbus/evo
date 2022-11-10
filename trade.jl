module Trade

export batch_reset!, batch_step!, PyTrade, render, get_metrics

using PyCall
using StatsBase
using Flux

#By default, PyCall doesn't include the current directory in the Python
#search path. If you want to do that (in order to load a Python module
#from the current directory), just run pushfirst!(pyimport("sys")."path",
#"").

pushfirst!(pyimport("sys")."path", "")
PyTrade = pyimport("trade_v4")

step_return_type = Tuple{PyDict{String,PyArray},  # obs
  Dict{String,Float32},   # rewards
  Dict{String,Bool}}      # Dones

function ecat(x...)
  @assert all(ndims(xi) == ndims(x[1]) for xi in x)
  cat(x..., dims=ndims(x[1]))
end

function reset!(env::PyObject)
  pycall(env.reset, PyDict{String,PyArray})
end

function render(env::PyObject, filename::String)
  pycall(env.render, Nothing, filename)
end

function get_metrics(env::PyObject)
  pycall(env.mc.return_metrics, PyDict{String,Float32}, env)
end
function step!(env::PyObject, actions::Dict{String,Int})::step_return_type
  pycall(env.step, step_return_type, actions)
end
function sample_batch(probs::Matrix{Float32})
  [sample(1:size(probs, 1), Weights(probs[:, i])) for i in 1:size(probs, 2)]
end

# TODO test that this works
# TODO figure out how/when to change agents
function reset!(env::PyObject, models::Dict{String,<:Chain})
  map(Flux.reset!, values(models))
  reset!(env)
end

function batch_dict(d::Vector{<:AbstractDict})
  Dict([key => ecat([di[key] for di in d]...) for key in keys(d[1])]...)
end

function batch_reset!(envs::Vector{PyObject}, models::Dict{String,<:Chain})
  obss = [reset!(env, models) for env in envs]
  @assert all(keys(obssi) == keys(obss[1]) for obssi in obss)
  batch_dict(obss)
end

# TODO test batch size 1 on virtual batch normalization

function batch_step!(envs::Vector{PyObject}, models::Dict{String,<:Chain}, obs::Dict{String,<:AbstractArray}; evaluation=false)
  @assert length(obs) == 1
  name, ob = first(obs)
  probs = models[name](ob) # bottleneck
  if evaluation
    # matrix of floats to matrix of cartesian indicies
    # to vector of cartesian indicies to vector of ints
    acts = argmax(probs, dims=1)[1, :] .|> z -> z[1]
    @infiltrate
  else
    acts = sample_batch(probs)
  end
  @assert length(acts) == length(envs)
  obss, rews, dones = Vector{PyDict{String,PyArray,true}}(), [], []
  for (env, act) in zip(envs, acts)
    obs, rew, done = step!(env, Dict(name => act)) # biggest bottleneck
    push!(obss, PyDict(obs))
    push!(rews, rew)
    push!(dones, done)
  end
  batch_dict(obss), rews, dones
end

end


# function test_single()
#   env = Trade.Trade(env_config)
#   obs_size = (env.obs_size..., 1)
#   num_actions = env.num_actions
#   # models = Dict("f0a0" => make_model(:small),
#   #   "f1a0" => make_model(:small, obs_size, num_actions))
#   obs = reset!(env, models)
#   o = obs["f0a0"]
#   o2 = [o;;;; o]
#   model = make_model(:small, size(o2), env.num_actions)
#   output = model(o2)
#   acts = sample_batch(output)
# end



