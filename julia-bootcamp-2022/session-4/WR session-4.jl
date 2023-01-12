test_function(x) = x + 1
@code_typed test_function(1.0)
@code_llvm test_function(1.0)
@code_native test_function(1.0)
@code_native debuginfo=:none test_function(1.0)
@code_native debuginfo=:none test_function(1)
@code_typed debuginfo=:none test_function('a')

#####
# type stability: an example
#####
bad_function(x) = x >= 5 ? 'a' : 5*2
@code_warntype bad_function(3)

function nested_function(y)
    x = bad_function(y)
    return x + 1
end
@code_warntype nested_function(1.0)

maybe_zero(x) = x >= 0 ? 0 : x
@code_warntype maybe_zero(1.0)

using BenchmarkTools
xs = randn(1_000)
@benchmark map(maybe_zero, $xs)

better_zero(x) = x >= 0 ? zero(x) : x
@code_warntype better_zero(1.0)
@benchmark map(better_zero, $xs)

#####
# Traceur.jl
#####
using Traceur
me_badman(x) = x <= 5 ? convert(typeof(x), 5) : x
@trace me_badman(5.0)

##### 
# Concrete typing
#####
arr = [12, "howdy partner", 3.9]
eltype(arr)
arr

# bad and slow array of reals
xs = randn(300, 300)
matrix = Matrix{Real}(xs)
@time matrix * matrix;
# good and fast array of Float64
matrix = Matrix{Float64}(xs)
@time matrix * matrix;

# an example of operationalizing
do_stuff(x) = (x=x, sin_x=sin(x), str_x=string(x))
@code_warntype do_stuff(1.0)
typeof(do_stuff(1))

function bad_arrays(xs)
    arr = []
    for x in xs
        push!(arr, do_stuff(x))
    end
    return mapreduce(nt -> nt.sin_x, +, arr)
end
function good_arrays(xs)
    first_val = do_stuff(xs[1])

    arr = [first_val]
    for x in xs[2:end]
        push!(arr, do_stuff(x))
    end
    return mapreduce(nt -> nt.sin_x, +, arr)
end
xs = randn(100)
b1 = @benchmark bad_arrays($xs)
b2 = @benchmark good_arrays($xs)
display(b1)
display(b2)
function best_arrays(xs)
    # map can do all the typing information for us
    arr = map(do_stuff, xs)
    return mapreduce(nt -> nt.sin_x, +, arr)
end
@benchmark best_arrays(xs)

#####
# Abstract types and structs
#####
struct StructWithField
    a::Float64
    b::Int
    c::String
end
struct BadlyTypedStruct
    a::Real
    b::Integer # abstract for Int
    c::Any
    d # equivalent to d::Any
end

#####
# Parametric types
#####
struct TypeName{TypeVariable}
    field::TypeVariable
end
struct ParametricStruct{A<:Real,B<:Integer,C}
    field_one::A
    field_two::B
    field_three::C
    field_four::A
end
ParametricStruct(1.0, 3, randn(10), 19.0)

#####
# Break up big functions into multiple smaller functions
#####
function fill_random(n)
    v = rand(Bool) ? Vector{Float64}(undef, n) : Vector{Int}(undef, n)
    for i in 1:n
        v[i] = rand(eltype(v))
    end
    return v
end
@benchmark fill_random(1000)

# rfill! can be specialized for Float64 or Int
function rfill!(v::Vector{T}) where T
    for i in eachindex(v)
        v[i] = rand(T)
    end
end
function fill_random_compartment(n)
    v = rand(Bool) ? Vector{Float64}(undef, n) : Vector{Int}(undef, n)
    rfill!(v)
    return v
end
@benchmark fill_random_compartment(1000)

#####
# Memory allocations
#####
using StaticArrays
function array_messabout(m)
    a = m*m'
    b = inv(a)
    return lu(b)
end
k = 10
m = randn(k,k)
m_static = SMatrix{k,k}(m)
@btime array_messabout($m);
@btime array_messabout($m_static);

struct Immutable{T}
    a::T
    b::T
    c::Matrix{T}
end
mutable struct Mutable{T}
    a::T
    b::T
    c::Matrix{T}
end
function double(m::Immutable)
    return Immutable(
        m.a * 2,
        m.b * 2,
        m.c * 2
    )
end
function double!(m::Mutable)
    m.a *= 2
    m.b *= 2
    m.c *= 2
    return m
end
double(m::Mutable) = double!(m)

function mult_double(val, n::Int) where T
    for i in 1:n
        val = double(val)
    end
    return val
end
imm = Immutable(1.0, 2.0, randn(100,100));
mut = Mutable(1.0, 2.0, randn(100,100));
@btime mult_double($imm, 1000);
@btime mult_double($mut, 1000);

#####
# column order v row order
#####
function rowsum(M)
    s = zero(eltype(M))
    for row in 1:size(M,1)
        for col in 1:size(M,2)
            s += M[row, col]
        end
    end
    return s
end
function colsum(M)
    s = zero(eltype(M))
    for col in 1:size(M,2)
        for row in 1:size(M,1)
            s += M[row, col]
        end
    end
    return s
end
M = randn(10_000, 10_000)
@btime rowsum($M);
@btime colsum($M);

#####
# @inbounds
#####
function sumprod_normal(A::AbstractArray, B::AbstractArray)
    @assert size(A) == size(B)
    r = zero(eltype(A))
    for i in eachindex(A)
        r += A[i] * B[i]
    end
    return r
end
function sumprod_inbounds(A::AbstractArray, B::AbstractArray)
    @assert size(A) == size(B)
    r = zero(eltype(A))
    for i in eachindex(A)
        @inbounds r += A[i] * B[i]
    end
    return r
end
xs1 = randn(10_000)
xs2 = randn(10_000)
@btime sumprod_normal(xs1,xs2)
@btime sumprod_inbounds(xs1,xs2)

#####
# @simd
#####
function sumprod_simd(A::AbstractArray, B::AbstractArray)
    @assert size(A) == size(B)
    r = zero(eltype(A))
    @simd for i in eachindex(A)
        r += A[i] * B[i]
    end
    return r
end
@btime sumprod_simd($xs1,$xs2);

function sumprod_super(A::AbstractArray, B::AbstractArray)
    @assert size(A) == size(B)
    r = zero(eltype(A))
    @simd for i in eachindex(A)
        @inbounds r+= A[i] * B[i]
    end
    return r
end;
@btime sumprod_super($xs1,$xs2);

#####
# In-place operations
#####
using LinearAlgebra
function eigtimer()
    M = randn(50,50)
    @btime eigvals(M)
    @btime eigvals!(M);
    return
end
# eigtimer()

#####
# Multiple dispatch
#####
function square(thing)
    if typeof(thing) <: Real
        return thing ^ 2
    elseif typeof(thing) <: Vector && eltype(thing) <: Real
        return thing * thing'
    end
end
@btime square(1.0)
@btime square([1.0, 2.0])