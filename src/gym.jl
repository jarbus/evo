module Gym
export make, step!, reset!
import Evo
using PyCall

function make(env_name::String, num_envs::Int64=1)
  gym = pyimport("gymnasium")
  py"""
  import gymnasium as gym
  def make(env_name):
    return gym.vector.SyncVectorEnv([
        lambda: gym.make(env_name) for i in range($num_envs)
    ])
  """
  env = py"make"(env_name)
  env.reset()
  env
end

Evo.reset!(env::PyObject) = Matrix(env.reset()[1]')
function Evo.step!(env::PyObject, actions::Vector{T}) where T <: Integer 
  observation, reward, terminated, truncated, info = env.step(actions)
  Matrix(observation'), reward, terminated, truncated, info
end


end
