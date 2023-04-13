module Gym
export make, step!, reset!
import Evo
using PyCall

function make(env_name::String, num_envs::Int64=1)
  gym = pyimport("gymnasium")
  env = gym.vector.make(env_name, num_envs)
  env.reset()
  env
end

Evo.reset!(env::PyObject) = env.reset()
Evo.step!(env::PyObject, actions::Vector{T}) where T <: Integer = env.step(actions)


end
