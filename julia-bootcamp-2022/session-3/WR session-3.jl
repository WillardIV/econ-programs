f(x) = sin(x^2)
df(x) = 2 * x * cos(x)

using Plots
# 100 values of x from 0,3
xs = range(0, 3, length = 100)
plot(xs, f, label = "f(x)")
plot!(xs, df, label = "df(x)")

####################
# Using autodiff
####################
# import Pkg; Pkg.add(["ReverseDiff", "ForwardDiff"])
import ReverseDiff, ForwardDiff
f(x) = sin(x^2)

exact_df(x) = 2*x*cos(x^2)
forward_df(x) = ForwardDiff.derivative(f, x)
forward_grad_df(x) = ForwardDiff.gradient(z -> f(z[1]), [x])[1]
reverse_df(x) = ReverseDiff.gradient(z -> f(z[1]), [x])[1]

@info "AD Examples" exact_df(1.0) reverse_df(1.0) forward_df(1.0)

xs = range(0, 3, length=100)
plot(xs, f, label="f(x)")
plot!(xs, exact_df, label="exact_df(x)")
plot!(xs, forward_df, label="forward_df(x)")
plot!(xs, reverse_df, label="reverse_df(x)")

# Concept check
using LinearAlgebra
n = 100
beta = [1.0, 2.2]
X = [ones(n) rand(n)]
Y = X*beta + randn(n)
beta_hat = inv(X'X)*X'Y;
f(b) = (Y - X*b)' * (Y - X*b)
df(b) = ReverseDiff.gradient(f, b)
df(beta_hat)

####################
# AD (usually) workds on any function
####################
using ForwardDiff, Distributions
choosy(arr) = choosy(arr...)
function choosy(x, mu, sigma)
    if x >= 1
        return logpdf(LogNormal(mu, sigma), x)
    elseif x <= -1
        return logpdf(Normal(mu, sigma), x)
    else
        return logpdf(Beta(15, 12), abs(x)) * sin(mu) * cos(sigma)
    end
end

println(ForwardDiff.gradient(choosy, [1.0, 2.0, 3.0]))
println(ForwardDiff.gradient(choosy, [0.5, 2.0, 3.0]))
println(ForwardDiff.gradient(choosy, [-1.2, 2.0, 3.0]))

function g_basic(x::Float64)
    return cos(sqrt(abs(sin(x^2))))
end
xs = range(0, 3, length = 100)
plot(xs, g_basic)
ForwardDiff.derivative(g_basic, 1.0)

function g(x) # alternatively function g(x::Real)
    return cos(sqrt(abs(sin(x^2))))
end
ForwardDiff.derivative(g, 1.0)

function g_rev(x::Float64)
    println(typeof(x))
    return cos(sqrt(abs(sin(x^2))))
end
ReverseDiff.gradient(m -> g_rev(m[1]), [1.0])][1]


####################
# In-place derivatives
####################
function fun_function(p)
    return sin(p[1]^2) * cos(p[2] - p[1]) + p[3]^3
end
# Version that allocates a return value each time
result = ForwardDiff.gradient(fun_function, [1.0, 2.1, 2.1])
# Version that places the result in an existing array
buf = zeros(3)
ForwardDiff.gradient!(buf, fun_function, [1.0, 2.1, 2.1])
# the results are the same
buf == result

rng = range(-5, 5, length = 100)
p_grid = vec(collect([a,b,c] for a in rng, b in rng, c in rng));
p_grid[1:3]

# Using ForwardDiff.gradient
function simple_grad(p_grid)
    grads = zeros(length(p_grid), 3)
    for i in 1:length(p_grid)
        grads[i,:] = ForwardDiff.gradient(fun_function, p_grid[i])
    end
    return grads
end

# Using ForwardDiff.gradient! with a buffer
function buf_grad(p_grid)
    grads = zeros(length(p_grid), 3)
    for i in 1:length(p_grid)
        # Note that I use view() here -- this gives the LOCATION of an 
        # array. Using grads[i,:] copies the array.
        ForwardDiff.gradient!(view(grads, i, :), fun_function, p_grid[i])
    end
    return grads
end

using BenchmarkTools
bm1 = @benchmark simple_grad(p_grid)
bm2 = @benchmark buf_grad(p_grid)
display(bm1)
display(bm2)

function big_function(p)
    n = div(length(p), 2)
    alphas = p[1:n]
    betas = p[n+1:end]

    return sum(alphas .* sin.(cos.(betas)))
end
xs_2 = randn(2)
xs_10 = randn(10)
xs_1k = randn(1000)
xs_10k = randn(10_000)

display(@benchmark ForwardDiff.gradient(big_function, xs_2))
display(@benchmark ReverseDiff.gradient(big_function, xs_2))
display(@benchmark ForwardDiff.gradient(big_function, xs_10))
display(@benchmark ReverseDiff.gradient(big_function, xs_10))

####################
# Finite differences
####################
# import Pkg; Pkg.add("FiniteDiff")
using FiniteDiff, LinearAlgebra, DataFrames
xs = rand(MvNormal([1.0, 2.0], I), 100)
mean_lp(mu) = sum(logpdf(MvNormal(mu, I), xs))
fd = FiniteDiff.finite_difference_gradient(mean_lp, [1.0, 2.0])
ad = ForwardDiff.gradient(mean_lp, [1.0, 2.0])
DataFrame(finite=fd, automatic=ad, difference=fd-ad)

bm1 = @benchmark FiniteDiff.finite_difference_gradient(mean_lp, [1.0, 2.0])
bm2 = @benchmark ForwardDiff.gradient(mean_lp, [1.0, 2.0])

display(bm1)
display(bm2)

####################
# Optim vs JuMP
####################
# import Pkg; Pkg.add("Optim")
plot(-5:0.1:5, x -> x^2)

using Optim
# Define target function
f(x) = x[1]^2
# Minmize with default (Nelder-Mead)
res = optimize(f, [1.0])
# Get minimizer info
@info "Nelder-Mead optimum" res.minimizer res.minimum Optim.converged(res)
println(res)

# Concept check
using Distributions, LinearAlgebra
n = 1000
means = [1.0, 0.0, 3.0, 4.0]
dist = MvNormal(means, I)
xs = rand(dist, n)
loglike(theta) = -sum(logpdf(MvNormal(theta, I), xs))

res = optimize(loglike, zeros(4), Optim.Options(iterations=10_000))
res.minimizer

res = optimize(f, [1.0], LBFGS())
@info "LBFGS optimum" res.minimizer res.minimum Optim.converged(res)

res = optimize(f, [1.0], LBFGS(), autodiff=:forward)
@info "Nelder-Mead optimum" res.minimizer res.minimum Optim.converged(res)

####################
# Using your own gradients
####################
# analytic gradient
function g!(G, x)
    G[1] = 2*x[1]
end
# minimize with LBFGS and exact gradient
res = optimize(f, g!, [1.0], LBFGS())
@info "Nelder-Mead optimum" res.minimizer res.mimimum Optim.converged(res)

using Distributions, LinearAlgebra, ReverseDiff
n = 100|0
K = 100
means = randn(K);
variances = rand(InverseGamma(2,3), K)
dist = MvNormal(means, Diagonal(variances))
xs = rand(dist, n);
function loglike(par)
    return -sum(logpdf(MvNormal(par[1:K], Diagonal(par[K+1:end])), xs))
end

# Bounds
lower = vcat(repeat([-Inf], K), repeat([0], K));
upper = vcat(repeat([Inf], 2K));
tape = ReverseDiff.GradientTape(loglike, zeros(2*K))
g!(G, x) = ReverseDiff.gradient!(G, tape, x)
res = optimize(loglike, g!, lower, upper, ones(K*2), Fminbox(LBFGS()))


####################
# JuMP
####################
# import Pkg; Pkg.add(["JuMP", "Clp", "HiGHS", "Ipopt"])
using JuMP
using Clp
model = Model(Clp.Optimizer)
@variable(model, x >= 0)
@variable(model, 0 <= y <= 3)
@objective(model, Min, 12x + 20y)
@constraint(model, c1, 6x + 8y >= 100)
@constraint(model, c2, 7x + 12y >= 120)
print(model)
optimize!(model)

# Show some stats on the solution
@show termination_status(model)
@show primal_status(model)
@show dual_status(model)
@show objective_value(model)
@show value(x)
@show value(y)
@show shadow_price(c1)
@show shadow_price(c2)

# Save solution to disk
write_to_file(model, "model.mps")

# Read from disk
model = read_from_file("model.mps")

# Checking solution status
model = Model(Clp.Optimizer)
@variable(model, 0 <= x <= 1)
@variable(model, 0 <= y <= 1)
@constraint(model, x + y >= 5)
@objective(model, Max, x - y)
optimize!(model)
termination_status(model) == INFEASIBLE

model = Model(Clp.Optimizer)
set_optimizer_attribute(model, "LogLevel", 4)
@variable(model, 0 <= x <= 1)
@variable(model, 0 <= y <= 1)
@constraint(model, x + y >= 5)
@objective(model, Max, x - y)
optimize!(model)
termination_status(model) == INFEASIBLE

# variables and domain constraints
model = Model()
@variable(model, x_free)
@variable(model, x_lower >= 0)
@variable(model, x_upper <= 1)
@variable(model, 2 <= x_interval <= 3)
@variable(model, x_fixed == 4)
print(model)

# vector/matrix variables
model = Model()
n = 10
@variable(model, x[1:n])
@variable(model, y[1:n, 1:n])
model

# vector constraints
model = Model()
bounds = rand(3)
@variable(model, sqrt(i) <= x[i = 1:3] <= i^2)
@variable(model, y[i = 1:3] <= bounds[i])

# integer/binary variables
model = Model()
@variable(model, x_binary, Bin)
@variable(model, x_int, Int)
@variable(model, 0 <= x_int_limited <= 5, Int)

# positive semidefinite and symmetric matrix variables
model = Model()
n = 10
@variable(model, Σ[1:n, 1:n], PSD)
@variable(model, Ω[1:n, 1:n], Symmetric)

# constraints
using Ipopt
model = Model(Ipopt.Optimizer)
@variable(model, x[1:5])
@variable(model, 0 <= q[1:5] <= 1)
@constraint(model, q'x <= 15)
@constraint(model, sum(q) == 1)
@constraint(model, q[5]*x[1] + 2*x[3] <= x[2] * x[5])
@objective(model, Max, q'x)
@NLconstraint(model, sqrt(sum(x[i]*q[i+1] for i in 1:2)) <= 100)
print(model)

# programmatic constraints
model = Model()
@variable(model, z[1:10])
x = randn(10)
for i in 1:length(x)
    if x[i] <= 0
        @constraint(model, 0 <= z[i])
    end
end
print(model)

# concept check
using Random; Random.seed!(15)
n_stocks = 10
μ = rand(MvNormal(zeros(n_stocks), 0.1I))
Σ = rand(InverseWishart(n_stocks, diagm(ones(n_stocks))))
function opt_portfolio(R)
    portfolio = Model(Ipopt.Optimizer)
    @variable(portfolio, -1 <= q[1:n_stocks] <= 1)
    @objective(portfolio, Min, q' * Σ * q)
    @constraint(portfolio, q'μ == R)
    @constraint(portfolio, sum(q) == 1)
    optimize!(portfolio)
    q_star = value.(q)

    return q_star'*Σ*q_star, q_star'*μ
end
rs = collect(range(-1, 1.0, length=100))
ys = map(opt_portfolio, rs)
using Plots
plot(ys)