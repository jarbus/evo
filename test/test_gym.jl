using Evo
using Flux
using Test
# make a test set to run through cartpole
@testset "CartPole" begin
  env = make("CartPole-v1")
  reset!(env)
  @test env.observation_space.shape == (1,4,)
  @test env.action_space.nvec[1] == 2
  for i in 1:10
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = step!(env, action)
    if any(terminated) || any(truncated)
       observation, info = env.reset()
     end
  end
end

@testset "RunGymBatch" begin
  models = Dict("test" => Chain(Dense(4, 2), softmax))
  env = make("CartPole-v1", 2)
  args = Dict("batch-size" => 1, "episode-length" => 10)
  run_batch(env, models, args; evaluation=false, render_str=nothing, batch_size=nothing)

end
