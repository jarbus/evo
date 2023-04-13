evologger = EvoLogger(stdout)

println("Testing logger")
with_logger(evologger) do
    @info "This is an info log message"
    @warn "This is a warning log message"
end


evologger = EvoLogger("test/test.log")

with_logger(evologger) do
    @info "This is an info log message"
    @warn "This is a warning log message"
end
