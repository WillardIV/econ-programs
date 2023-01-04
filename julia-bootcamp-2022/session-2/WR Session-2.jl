# using Pkg; Pkg.add("BenchmarkTools")
using LinearAlgebra, BenchmarkTools

function expensive()
    X = randn(10_000, 100)
    return inv(X'X)
end

function go()
    A = expensive()
    B = expensive() # Have to wait for the first call to finish!
    return A + B
end

# @benchmark runs expensive a bunch of times to estimate
# the computational run time.
@benchmark go()

# usign LinearAlgebra
b() = begin
    sleep(5) # wait 5 seconds
    dot(randn(1000,1000), rand(1000,1000))
end

# Assign running b to a thread
task = Threads.@spawn b()
println("Waiting...")
println(fetch(task))

function expensive()
    X = randn(10_000, 1_000)
    return inv(X'X)
end
function go_serial()
    A = expensive()
    B = expensive()

    return A + B
end
function go_threads()
    A = Threads.@spawn expensive()
    B = Threads.@spawn expensive()

    # Wait for both to complete
    return fetch(A) + fetch(B)
end

@benchmark go_serial()
@benchmark go_threads()

function thread_test()
    Threads.@threads for i in 1:6
        id = Threads.threadid()
        println("Thread ID: ", id, " | i: ", i)
    end
end
thread_test()