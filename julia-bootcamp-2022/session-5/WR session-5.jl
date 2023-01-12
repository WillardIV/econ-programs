using Distributions, StatsPlots
params = [
    (1,1),
    (3,1),
    (3,6),
    (10,10)
]
ps = plot();
for (alpha, beta) in params
    plot!(ps, Beta(alpha, beta), label = "α=$alpha, β=$beta")
end
display(ps)

ps = plot();
for i in [1, 10, 50, 100, 200]
    plot!(ps, Beta(i,i), label="α=$i, \beta=$i")
end
ps


# true heads probability
true_prob = 0.45
# coin flips
n_draws = 10
data = rand(Binomial(1,true_prob), n_draws);
histogram(data)

# prior parameters
α0 = β0 = 10
# posterior density function
function posterior_coinflip(α0, β0, x)
    α = α0 + sum(x)
    β = β0 + length(x) - sum(x)
    return α, β
end
α, β = posterior_coinflip(α0, β0, data)
# show prior and posterior
prior_dist = Beta(α0, β0)
posterior_dist = Beta(α, β)
# plot it
plot(prior_dist, label="Prior")
plot!(posterior_dist, label="Posterior")

posteriors = Distribution[]
plt = plot(prior_dist, label="Prior")
data_counts = [1, 10, 100, 1_000, 10_000, 100_000]
for n_data in data_counts
    data = rand(Binomial(1,true_prob), n_data)
    α, β = posterior_coinflip(α0, β0, data)
    post_dist = Beta(α, β)
    push!(posteriors, post_dist)
    plot!(post_dist, label="Posterior (n=$n_data)")
end
display(plt)

using DataFrames
summary_stats = DataFrame(
    n_trials = data_counts,
    true_prob = true_prob,
    mean = mean.(posteriors),
    var = var.(posteriors),
    alpha = map(x -> x.α, posteriors),
    beta = map(x -> x.β, posteriors),
    p05 = quantile.(posteriors, [0.05]),
    p95 = quantile.(posteriors, [0.95]),
);
display(summary_stats)

#####
# Markov chain Monte-Carlo
#####
using AdvancedMH
using Distributions
using LinearAlgebra
using MCMCChains
# generate data
data = rand(Normal(0,1), 30);
# define model
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], θ[2])
function density(θ)
    # define prior for mean and variance
    mu_prior = Normal(-1,2)
    sigma_prior = InverseGamma(2,3)
    # calculate likelihood + prior
    if insupport(θ)
        return sum(logpdf.(dist(θ), data)) +
            logpdf(mu_prior, θ[1]) +
            logpdf(sigma_prior, θ[2])
    else
        return -Inf
    end
end;
