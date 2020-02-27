# We will use unit testing to make sure our solutions are what we expect
# This shows how to import the Test package, which provides convenient functions like @test
using Test
# Setting a Random Seed is good practice so our code is consistent between runs
using Random # Import Random Package
Random.seed!(414); #Set Random Seed
# ; suppresses output, makes the writeup slightly cleaner.
# ! is a julia convention to indicate the function mutates a global state.
using Statistics
# Probability
#Q1
#1.1.1 [3pts] Starting from the definition of independence, show that the independence of X
# and Y implies that their covariance is 0.
#import Pkg; Pkg.add("Distributions")



function gaussian_pdf(x; mean=0., variance=0.01)
    #x = randn()
    f(x) = (1/(sqrt(2*pi*variance)))*(exp(-0.5((x-mean)^2)/(variance)))
    return f(x)
    #: implement pdf at x
end
#Test your implementation against a standard implementation from a library:
# Test answers
using Distributions: pdf, Normal # Note Normal uses N(mean, stddev) for parameters
@testset "Implementation of Gaussian pdf" begin
    x = randn()
    @test gaussian_pdf(x) ≈ pdf.(Normal(0.,sqrt(0.01)),x)
# ≈ is syntax sugar for isapprox, typed with `\approx <TAB>`
# or use the full function, like below
    @test isapprox(gaussian_pdf(x,mean=10., variance=1) , pdf.(Normal(10., sqrt(1)),x))
end


function sample_gaussian(n; mean=0., variance=0.01) # n samples from standard gaussian
    x = randn(n)
    a = sqrt(variance)
    b = mean
    z = a.*x.+b
    #stats = [mean(z) std(z) var(z)]
    return z
end

using Statistics: mean, var
@testset "Numerically testing Gaussian Sample Statistics" begin
# Sample 100000 samples with your function and use mean and var to
# compute statistics.
    z1 = sample_gaussian(10000)
# tests should compare statistics against the true mean and variance from
#arguments.

    @test isapprox(mean(z1), 0., atol=1e-2)
    @test isapprox(var(z1), 0.01, atol=1e-2)

# hint: use isapprox with keyword argument atol=1e-2
end;
Pkg.add("Plots")
Pkg.add("StatsPlots")
using Plots

#histogram(#TODO)
histogram(sample_gaussian(10000; mean=10, variance=2), normed=true)
#plot!(#TODO)


plot!(histogram(sample_gaussian(10000; mean=10, variance=2), normed=true),
(Normal(10,sqrt(2))))

savefig("A0plot")

Pkg.add("Zygote")
Core.AbstractArray

# Choose dimensions of toy data




m = rand(1:100)
n = rand(1:100)
# Make random toy data with correct dimensions
x = rand(1:1000,m)
y = rand(1:1000,m)
A = randn(m,n)
B = randn(m,m)

# Make sure your toy data is the size you expect!
@testset "Sizes of Toy Data" begin
#TODO: confirm sizes for toy data x,y,A,B
    @test size(A) == (m, n)
    @test size(B) == (m, m)
    @test size(x) == (m,)
    @test size(y) == (m,)
#hint: use `size` function, which returns tuple of integers.
end;

# Use AD Tool
using Zygote: gradient
using LinearAlgebra
# note: `Zygote.gradient` returns a tuple of gradients, one for each argument.
# if you want just the first element you will need to index into the tuple with [1]
#f1(x) = transpose(x)*y
#df1dx = gradient(x->f1(x)[1],x)

f1(x) = dot(transpose(vec(x)),vec(y))
df1dx = gradient(f1,x)

#f2(x) = transpose(x)*x
#df2dx = gradient(x->f2(x)[1],x)

f2(x) = dot(transpose(vec(x)),vec(x))
df2dx = gradient(f2,x)



f3(x) = ((vec(x)')*A)
df3dx = jacobian(f3,x)

df3dx == A'

function jacobian(f, x)
    y = f(x)
    n = length(y)
    m = length(x)
    T = eltype(y)
    j = Array{T, 2}(undef, n, m)
    for i in 1:n
        j[i, :] .= gradient(x -> f(x)[i], x)[1]
    end
return j
end

f4(x) = ((vec(x)')*B*vec(x))
df4dx = jacobian(f4,x)



@testset "AD matches hand-derived gradients" begin
    @test df1dx == (vec(y)',)
    @test df2dx == (vec(2x)',)
    @test df3dx == A'
    @test round.(df4dx, digits=10) == round.(x'*(B'+ B),digits=10)
    #@test df4dx == #
end;
  
