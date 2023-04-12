module Gym
export make, step!, reset!
import Evo
using PyCall

function make(env_name::String)
  gym = pyimport("gymnasium")
  env = gym.make(env_name)
  env.reset()
  env
end


Evo.reset!(env::PyObject) = env.reset()
Evo.step!(env::PyObject, action::Int64) = env.step(action)


end
