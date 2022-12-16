mk_env_config(args) = Dict(
    "window" => args["window"],
    "grid" => (args["gx"], args["gy"]),
    "food_types" => 2,
    "latest_agent_ids" => [0, 0],
    "matchups" => [("f0a0", "f1a0")],
    "episode_length" => args["episode-length"],
    "respawn" => true,
    "fires" => [Tuple(args["fires"][i:i+1]) for i in 1:2:length(args["fires"])],
    "foods" => [Tuple(args["foods"][i:i+2]) for i in 1:3:length(args["foods"])],
    "health_baseline" => true,
    "pickup_coeff" => args["pickup-coeff"],
    "light_coeff" => args["light-coeff"],
    "spawn_agents" => "center",
    "spawn_food" => "corner",
    "food_agent_start" => args["food-agent-start"],
    "food_env_spawn" => args["food-env-spawn"],
    "day_night_cycle" => true,
    "day_steps" => args["day-steps"],
    "seed" => args["seed"],
    "vocab_size" => 0)

# logs to stdout if islocal is true, else logs to name
function llog(f; islocal::Bool, name::String)
    logfile = !islocal ? open(name, "a") : stdout
    f(logfile)
    !islocal && close(logfile)
end

update_df(df::Nothing, mets::Nothing) = nothing
update_df(df::Nothing, mets)   = DataFrame(mets)
update_df(df::DataFrame, mets) = push!(df, mets)
write_mets(file_name::String, df::Nothing) = nothing
write_mets(file_name::String, df::AbstractDataFrame) = CSV.write(file_name, df)

ts(x) = println(Dates.format(now(), "HH:MM:SS")*" $x")
ts(f, x) = println(f, Dates.format(now(), "HH:MM:SS")*" $x")


