println("Hello World!")

x = 2
println(x + 2)

x = 5 + 8
println(x)

y = "howdy partner"

x1 = 15
red5 = 91
hats = 2
mapsie = -1
tuba = 9
answer = (x1 + red5) ^ hats * (mapsie/tuba)
println(answer)

####################
# PRIMITIVE TYPES
####################
typeof(5)
typeof(5.0)
typeof(true)
typeof("howdy!")
typeof('h')

# Concept check:
z = 2.368
println(z % 2)
println(z % 4)
println(z % 8)
quotients = [2,4,8]
map(x -> z % x, quotients)

####################
# Boolean operators
####################

# Concept check:
some_number = 16
if some_number >= 15
    println("It's larger than 15")
end

var_a = 15
var_b = 7.69
if var_a > 1 || !(var_b < 2)
    println("It's complicated")
end


####################
# Functions
####################
function f(x,y)
    x + y
end

function poly(x, y)
    return 3x^2 + 2y^2 - 2*x*y
end
poly(1, 2.5)

# Concept check:
function g(x, theta1, theta2, theta3, theta4)
    theta1*x^3 + theta2*x^2 + theta3*x + theta4
end
f(x, theta1, theta2, theta3, theta4) = theta1*x^3 + theta2*x^2 + theta3*x + theta4

####################
# Type annotations
####################
afun(x::Int) = x + 1
afun(x::Float64) = x / 2
afun(1)
afun(1.0)

function aces(x)
    if x > 5
        # This exits the function immediately
        return x + 5

        # This line will never run
        println(x - 9)
    end
        # If x <= 5, this line will run instead
        return x / 5
end

aces(3)
aces(8)

function noreturn(a)
    b = a % 2

    b # equivalent to writing return b
end

function noreturn(a)
    b = a % 2
    
    b # now this line is useless
    a
end

twovals(a,b) = (a*b, a/b, a+b, a-b)
twovals(1,2)

# Tuples
a_tuple = (5, 10, 15)
a_tuple[1]
a_tuple[3]

x = (1,2,3); x[1] = 4 # this won't run

a,b,c = (15, "red", 8)
a,b... = (15, "red", 7) # partial unpacking the a_tuple
println(typeof(b))

# Named Tuples
namedtup = (thing1 = 'a', thing2 = 15, thing3 = [10,9,8])
tup2 = (willard="awesome",)
merge(namedtup, tup2)
id = :thing2
namedtup[id]
namedtup.thing2 # equivalent

# Concept check:
function f(x)
    if x > 0
        return (2*x, "positive")
    elseif x < 0
        return (x*x, "negative")
    else
        return(0, "zero")
    end
end
f(1)
f(-4)
f(0)

####################
# Keywords
####################
f(x; coef = 2.0) = x * coef
f(2)
f(2, coef=10)

####################
# Array types
####################
vector_var = [5,10,19]
matrix_var = [1 2 3; 4 5 6]

# push!(vector, thing)
xs = [5,6,7]
push!(xs,10)
println(xs)

xs = [50, 100, 150]
println(xs[1])
M = [10 11 12; 20 21 22]
M[2,2]

xs = zeros(5)
xs[1] = 3
xs[2:3] = [4, 4.5]
xs[4:end] = [1.0, 2.0]
xs

# map
# map(function_name, things_to_input)
xs = [5, 10, 11]
doubler(input) = 2*input
map(doubler, xs)
doubler.(xs) # equivalent

map(doubler, 10:20)

xs = [5, 10, 11]
map(x -> 2*x, xs)

two_input_function(x,y) = x^2 / y
target(m) = two_input_function(m, 3)
map(target, [1,2,3,4])
map(m -> two_input_function(m, 3), [1,2,3,4])

inputs = [(1,2), (3,4), (5,6)]
map(m -> two_input_function(m[1], m[2]), inputs)

# Broadcasting
a = [1.0, 2.0, 3.0]
sin.(a)

# Concept check:
# Type "] add UnicodePlots" in the terminal
function quadratic(x)
    return x^2
end

xs = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
ys = map(x -> quadratic(x), xs)

using UnicodePlots
display(lineplot(xs,ys))

####################
# Control flow
####################
for i in [1,2,3,4,5] # 1:5
    println(i)
end

# Concept check:
function times_table(n)
    mat = zeros(Int,n,n)
    for i in 1:n
        for j in 1:n
            mat[i,j] = i*j
        end
    end
    return mat
end
println(times_table(4))

####################
# The while loop
####################
x = 100
while x > 1
    println(x)
    x = x / 2 # equivalently x /= 2
end
x

for i in 1:10
    println(i)
    break
end
println("all done")

for i in 1:10
    # Skip even numbers
    if rem(i, 2) == 0
        continue
    end
    println(i)
end
for i in 1:10
    if rem(i,2) != 0
        println(i)
    end
end

try
    hats_are_a_dogs_best_friend(15, "red")
catch e
finally 
    println("I did the work, boss")
end

####################
# Linear Algebra
####################
using LinearAlgebra

# Create a 100x3 matrix of standard normals
X = randn(100,3)
X[:,1] = ones(size(X,1))

# Define coefficients
β = [5.0, 2.0, -1.5]
Y = X * β + randn(size(X,1))

# Matrix squaring
X'X

# OLS normal equations
β_hat = inv(X'X)X'Y

# Making predictions using β_hat
Y_hat = X*β_hat

# Calculating the SSE
e = Y_hat - Y
sse = e'e

# Concept check:
n,K = size(X)
function ols_stats(X,Y)
    n,K = size(X)
    beta_hat = inv(X'X)X'Y
    Y_hat = X*beta_hat
    e = Y_hat - Y
    s2 = e'e/(n-K)
    vcov = s2 * inv(X'X)
    ts = zeros(K)
    for i = 1:K
        ts[i] = beta_hat[i] / vcov[i,i]
    end

    return (beta_hat, vcov, ts)
end
beta_hat, sigma_se, t = ols_stats(X,Y)
beta_hat
sigma_se
t

####################
# types
####################
# the <: operator says "left is a subtype of right"
Int64 <: Signed <: Integer <: Real <: Number <: Any

"howdy"::String
1::Float64 # error because 1 is an int
1::Real # real is a more general type

struct SomeStuff
    field::Float64
end
some_stuff = SomeStuff(0.5)
some_stuff.field

module VehicleTypes

# top level type
abstract type MobilityDevice end
# Subtypes
abstract type Car <: MobilityDevice end
abstract type Bicycle <: MobilityDevice end
# Subtypes for Car
struct Combustion <: Car
    gallons::Float64
    mpg::Float64
end
struct Electric <: Car
    emiles::Int64
end
# Subtypes for Bicycle
struct Road <: Bicycle
    purchase_cost::Float64
end
struct Mountain <: Bicycle
    maintenance_cost::Float64
end
# Make types available outside module
export Electric, Combustion,
    Road, Mountain,
    Car, Bicycle,
    MobilityDevice
end

# Import the types we just made
using Main.VehicleTypes
electric_car = Electric(225.0)
ice_car = Combustion(11.0, 21.0)
fast_bike = Road(1500.99)
cool_bike = Mountain(49.99)

# Which ones move?
moves(thing::MobilityDevice) = true
moves(thing::Car) = "hell yeah homie"
moves(thing) = false # equivalent to moves(thing::Any)
map(moves, [electric_car, ice_car, fast_bike, cool_bike, "hats"])

# Check which ones are human powered
human_powered(thing::Car) = false
human_powered(thing::Bicycle) = true
map(human_powered, [electric_car, ice_car, fast_bike, cool_bike])

# How far can we go?
distance_avail(thing::Electric) = thing.emiles
distance_avail(thing::Combustion) = thing.gallons * thing.mpg
map(distance_avail, [electric_car, ice_car])

# Which car should I buy?
function which_car(cars::Vector{<:Car})
    distances = map(distance_avail, cars)
    dist, position = findmax(distances)
    return cars[position]
end

cars = [
    Combustion(10,15),
    Combustion(13,12),
    Combustion(11,32),
    Electric(222),
    Electric(224),
    Electric(312)
]
best_car = which_car(cars)


####################
# Multiple dispatch
####################
println("Multiple dispatch is exemplified above")

####################
# Data input/output(I/O)
####################]
# ] add CSV DataFrames in repl
import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/cpfiffer/julia-bootcamp-2022/main/session-1/example.csv",
    "example.csv"
)

using CSV, DataFrames
df = DataFrame(CSV.File("example.csv"))
df.name
df[:, :name]
df[:, [:name, :attribute]]
select(df, [:name, :attribute]) # equivalent to above
df[1, :]
df[1:2, :]
df[df.awesomeness .>= 3, :]
df[df.owns_hat, :]
filter(x -> x.owns_hat, df) # equivalent to above

# Use groupby to get average wealth by hat ownership
using Statistics
group = groupby(df, :owns_hat)
combine(group, :wealth => mean, :awesomeness => mean, :awesomeness => std)

# Make a second dataframe with names & instead
name_df = DataFrame(
    name = ["jane", "emilio", "bojack"],
    id = ["J876-A", "E993-B", "B261-L"]
)
leftjoin(df, name_df, on=:name) # 2 columns and 4 rows omitted
rightjoin(df, name_df, on=:name) # 3 columns and 2 rows omitted
innerjoin(df, name_df, on=:name) # 2 columns omitted
outerjoin(df, name_df, on=:name) # 3 columns 5 rows omitted

# Make up some data
ndf = DataFrame(
    randn(1_000, 5),             # 1,000 rows, 5 columns of standard normals
    ["a", "b", "c", "d", "e"]   # column names
)
first(ndf, 3) # 1 column and 2 rows omitted
# Write file to disk
CSV.write("example-write.csv", ndf)


####################
# Visualization
####################
using Plots
xs = 0:0.1:15
ys = sin.(xs)
plot(xs,ys)

# We can add titles/legengs/colors/etc. with keywords:
plot(xs, ys, color=:red, title="A plot", label="series 1")

# Plot multiple series on top of each other
ys1 = sin.(xs)
ys2 = cos.(xs)
plot(xs, ys1)
plot!(xs, ys2)

plot(
    scatter(xs, ys1),
    plot(xs, ys1)
)

# overlay two plot types on top of each other
plot(xs, ys1)
scatter!(xs, ys1)

using StatsPlots
draws = randn(1000)
plot(
    density(draws),
    Plots.histogram(draws)
)

# Plots with marginals
using Distributions, StatsPlots
dist = MvNormal([0,0], [1.0 0.5; 0.5 2.0])
draws = rand(dist, 1000)
marginalkde(draws[1,:], draws[2,:])

using Distributions, StatsPlots
dist = MvNormal(
    [0,0,0],
    [
        1.0 0.5 -0.2;
        0.5 2.0 0.0;
        -0.2 0.0 4.0
    ]
)
draws = rand(dist, 1000)
scatter3d(draws[1,:], draws[2,:], draws[3,:], alpha = 0.5)


####################
# Project
####################
using LinearAlgebra, Statistics, DataFrames
n = 100
x = [ones(n) rand(1:0.1:10, n)]
b, sigma = [2.5, -1.2], 1
y = x*b + sigma .* randn(n);

# Define types
abstract type StandardErrorEstimator end
struct Bootstrap <: StandardErrorEstimator
    bootstrap_samples::Int
    subsample_size::Int
end
struct Spherical <: StandardErrorEstimator end

function ols(x, y, est::Spherical)
    n,k = size(x)
    beta_hat = inv(x'x)x'y
    yhat = x * beta_hat
    e = yhat - y
    s2 = e'e/(n-k)
    vcov = s2*inv(x'x)
    se = diag(vcov)

    return beta_hat, se
end
function ols(x, y, se_estimator::Bootstrap)
    n,k = size(x)
    n_samples = se_estimator.bootstrap_samples
    subsample_size = se_estimator.subsample_size
    beta_hats = zeros(n_samples, k)
    s2s = zeros(n_samples, k)
    for t = 1:subsample_size
        inds = rand(1:n, subsample_size)
        xt = x[inds, :]
        yt = y[inds]
        beta_hats[t, :] = inv(xt'xt)xt'yt
    end
    beta_hat = zeros(k, 1)
    se_hat = zeros(k, 1)
    for i in 1:k
        beta_hat[i] = mean(beta_hats[:,i])
        se_hat[i] = std(beta_hats[:,i])
    end

    return beta_hat, se_hat
end

T = 10000
m = 10

spherical = ols(x,y,Spherical())
bootstrap = ols(x,y,Bootstrap(T,m))
