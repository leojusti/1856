
# import Automatic Differentiation
# You may use Neural Network Framework, but only for building MLPs
# i.e. no fancy probabilistic implementations
using Flux
using MLDatasets
using Statistics
using Logging
using Test
using Random
using StatsFuns: log1pexp
Random.seed!(412414);

#### Probability Stuff
# Make sure you test these against a standard implementation!

# log-pdf of x under Factorized or Diagonal Gaussian N(x|μ,σI)
function factorized_gaussian_log_density(mu, logsig,xs)
  """
  mu and logsig either same size as x in batch or same as whole batch
  returns a 1 x batchsize array of likelihoods
  """
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((xs .- mu).^2)./(σ.^2),dims=1)
end


# log-pdf of x under Bernoulli
function bernoulli_log_density(logit_means,x)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
  b = x .* 2 .- 1 # {0,1} -> {-1,1}
  return - log1pexp.(-b .* logit_means)
end

bernoulli_log_density(randn(7,3),3)

## This is really bernoulli
@testset "test stable bernoulli" begin
  using Distributions
  x = rand(10,100) .> 0.5
  μ = rand(10)
  logit_μ = log.(μ./(1 .- μ))
  @test logpdf.(Bernoulli.(μ),x) ≈ bernoulli_log_density(logit_μ,x)
  # over i.i.d. batch
  @test sum(logpdf.(Bernoulli.(μ),x),dims=1) ≈ sum(bernoulli_log_density(logit_μ,x),dims=1)
end

# sample from Diagonal Gaussian x~N(μ,σI) (hint: use reparameterization trick here)
sample_diag_gaussian(μ,logσ) = (ϵ = randn(size(μ)); μ .+ exp.(logσ).*ϵ)
# sample from Bernoulli (this can just be supplied by library)
sample_bernoulli(θ) = rand.(Bernoulli.(θ))



# Load MNIST data, binarise it, split into train and test sets (10000 each) and partition train into mini-batches of M=100.
# You may use the utilities from A2, or dataloaders provided by a framework
function load_binarized_mnist(train_size=10000, test_size=1000)
  train_x, train_label = MNIST.traindata(1:train_size);
  test_x, test_label = MNIST.testdata(1:test_size);
  @info "Loaded MNIST digits with dimensionality $(size(train_x))"
  train_x = reshape(train_x, 28*28,:)
  test_x = reshape(test_x, 28*28,:)
  @info "Reshaped MNIST digits to vectors, dimensionality $(size(train_x))"
  train_x = train_x .> 0.5; #binarize
  test_x = test_x .> 0.5; #binarize
  @info "Binarized the pixels"
  return (train_x, train_label), (test_x, test_label)
end

function batch_data((x,label)::Tuple, batch_size=100)
  """
  Shuffle both data and image and put into batches
  """
  N = size(x)[end] # number of examples in set
  rand_idx = shuffle(1:N) # randomly shuffle batch elements
  batch_idx = Iterators.partition(rand_idx,batch_size) # split into batches
  batch_x = [x[:,i] for i in batch_idx]
  batch_label = [label[i] for i in batch_idx]
  return zip(batch_x, batch_label)
end
# if you only want to batch xs
batch_x(x::AbstractArray, batch_size=200) = first.(batch_data((x,zeros(size(x)[end])),batch_size))


#batch_data((randn(13),3)::Tuple, 100)

### Implementing the model

## Load the Data
train_data, test_data = load_binarized_mnist();
train_x, train_label = train_data;
test_x, test_label = test_data;

## Test the dimensions of loaded data
@testset "correct dimensions" begin
@test size(train_x) == (784,10000)
@test size(train_label) == (10000,)
@test size(test_x) == (784,1000)
@test size(test_label) == (1000,)
end

## Model Dimensionality
# #### Set up model according to Appendix C (using Bernoulli decoder for Binarized MNIST)
# Set latent dimensionality=2 and number of hidden units=500.
Dz, Dh = 2, 500
Ddata = 28^2

# ## Generative Model
# This will require implementing a simple MLP neural network
# See example_flux_model.jl for inspiration
# Further, you should read the Basics section of the Flux.jl documentation
# https://fluxml.ai/Flux.jl/stable/models/basics/
# that goes over the simple functions you will use.
# You will see that there's nothing magical going on inside these neural network libraries
# and when you implemented a neural network in previous assignments you did most of the work.
# If you want more information about how to use the functions from Flux, you can always reference
# the internal docs for each function by typing `?` into the REPL:
# ? Chain
# ? Dense



#  QUESTION 1

######## A
## Model Distributions
log_prior(z) = factorized_gaussian_log_density(0,0,z)

####### B

# create multi layer perceptron network with 500 dim hidden layer and tanh
# decoder outputs logit means
# Z X H -> H X D
decoder2 = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2))


#decoder_function(rand(2,1))

####### C

function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z)"""
  # z = decoder zs
  θ = decoder2(z)
  return sum(bernoulli_log_density(θ,x), dims=1)
  # return likelihood for each element in batch
end

####### D

joint_log_density(x,z) = log_prior(z) .+ log_likelihood(x,z)


##### QUESTION 2

function factorized_gaussian_log_density(mu, logsig,xs)
  """
  mu and logsig either same size as x in batch or same as whole batch
  returns a 1 x batchsize array of likelihoods
  """
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((xs .- mu).^2)./(σ.^2),dims=1)
end

# log-pdf of x under Bernoulli
function bernoulli_log_density(logit_means,x)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
  b = x .* 2 .- 1 # {0,1} -> {-1,1}
  return - log1pexp.(-b .* logit_means)
end


## Amortized Inference
function unpack_gaussian_params(θ)
  μ, logσ = θ[1:2,:], θ[3:end,:]
  return  μ, logσ
end


###     A
# D X H -> H X Z
encoder = Chain(Dense(Ddata,Dh, tanh), Dense(Dh, Dz*2))
# Hint: last "layer" in Chain can be 'unpack_gaussian_params'

###     B
log_q(q_μ, q_logσ, z) = factorized_gaussian_log_density(q_μ,q_logσ,z)


###     C
function elbo(x)
  encoder_params = encoder(x)
  q_μ, q_logσ = unpack_gaussian_params(encoder_params)
  z = sample_diag_gaussian(q_μ,q_logσ)
  joint_ll = joint_log_density(x,z)
  log_q_z = log_q(q_μ, q_logσ,z)
  elbo_estimate = mean(joint_ll - log_q_z)
  return elbo_estimate
end

###     D
function loss(x)
  loss = -elbo(x)
  return loss
end

# See example_flux_model.jl for inspiration

###     E
function train_model_params!(loss, encoder, decoder, train_x, test_x; nepochs=10)
  # model params
  ps = Flux.params([encoder,decoder])
  # ADAM optimizer with default parameters
  opt = ADAM()
  # over batches of the data
  for i in 1:nepochs
    for d in batch_x(train_x,200)
      gs = Flux.gradient(()-> loss(d),ps)
      Flux.Optimise.update!(opt,ps,gs)
    end
    if i%1 == 0 # change 1 to higher number to compute and print less frequently
      @info "Test loss at epoch $i: $(loss(batch_x(test_x)[1]))"
    end
  end
  @info "Parameters of encoder and decoder trained!"
end


## Train the model
train_model_params!(loss,encoder,decoder2,train_x,test_x, nepochs=100)

Pkg.add("BSON")
### Save the trained model!
using BSON:@save
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
save_dir = "trained_models"
if !(isdir(save_dir))
  mkdir(save_dir)
  @info "Created save directory $save_dir"
end
@save joinpath(save_dir,"encoder_params.bson") encoder
@save joinpath(save_dir,"decoder_params.bson") decoder2
@info "Saved model params in $save_dir"



## Load the trained model!
using BSON:@load
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
load_dir = "trained_models"
@load joinpath(load_dir,"encoder_params.bson") encoder
@load joinpath(load_dir,"decoder_params.bson") decoder2
@info "Load model params from $load_dir"

# QUESTION 3

# cd("/Users/justinleo/Documents/GitHub/STA414-2020-A3-leojusti/")
# Pkg.add("Images")
# Pkg.add("ImageMagick")
# Pkg.add("QuartzImageIO")
# Pkg.add("ImageIO")

# Visualization
using Images
using Plots
# make vector of digits into images, works on batches also
mnist_img_alt(x) = ndims(x)==2 ? Gray.(permutedims(reshape(x,28,28,:), [2, 1, 3])) : Gray.(transpose(reshape(x,28,28)))
## Example for how to use mnist_img to plot digit from training data
plot(mnist_img_alt(train_x[:,1]))

######      A.1
z_10 = [sample_diag_gaussian(0,0),sample_diag_gaussian(0,0)]

#####       A.2
d10 = decoder2(z_10)
u = exp.(decoder2(z_10)) #bernoulli means
p = u ./ (1 .+ u)

image = mnist_img_alt(p)
plot(image, title="Bernoulli Means Over Pixels")
savefig("Q3a.2.pdf")


#####       A.3
#plot()

B = sample_bernoulli(p)
image2 = mnist_img_alt(B)
plot(image2, title="Binary Image x from Sample of Bernoullis")
savefig("Q3a.3.1.pdf")

sampleR = randn(2,10)
d20 = decoder2(sampleR)
u2 = exp.(decoder2(sampleR))
p2 = u2 ./ (u2.+1)
image_test = mnist_img_alt(d20)
sample_u = sample_bernoulli(p2)
image_sample = mnist_img_alt(sample_u)

fst = hcat(mnist_img_alt.([p2[:, i] for i in 1:10])...)
snd = hcat(mnist_img_alt.([sample_u[:, i] for i in 1:10])...)


plot(vcat(fst,snd),
      xaxis=false, yaxis=false, titlefontsize = 7, titlefontcolor =:black , title = "10 Samples of Bernoulli Means vs. Binarized Image")
      plot!(size=(320,110))
savefig("Q3a.3.2.pdf")

#### B1

encoder(train_x)


#### B2

e = unpack_gaussian_params(encoder(train_x))[1]


#### B3


plot(e[1,:],e[2,:], seriestype = :scatter,  group =train_label,
      title = "Mean Vectors in 2D Latent Space", xlabel = "Mean",  xlim=(-6,6), ylabel = "Mean", legend=:topright)
savefig("Q3Bc.pdf")
### C1

function convex_weighting(za,zb,α)
  if 0<=α<=1
    return(za_new = α*(za) + (1-α)*(zb))
  else
    return("α not valid")
  end

end

convex_weighting(1,2,0.3)

### C2
sample_pairing_1 = train_x[:,1]
sample_pairing_2 = train_x[:,2]
sample_pairing_3 = train_x[:,3]

### C3
zs1 = unpack_gaussian_params(encoder(sample_pairing_1))
zs2 = unpack_gaussian_params(encoder(sample_pairing_2))
zs3 = unpack_gaussian_params(encoder(sample_pairing_3))

### C4
c1 = convex_weighting(zs1[1],zs2[1], rand())
c2 = convex_weighting(zs1[1],zs3[2], rand())
c3 = convex_weighting(zs2[1],zs3[2], rand())

θ1 = decoder2(c1) #bern means
θ2 = decoder2(c2)
θ3 = decoder2(c3)

hcat(mnist_img_alt(reshape(θ1, length(θ1),)),
      mnist_img_alt(reshape(θ2, length(θ2),)),
      mnist_img_alt(reshape(θ3, length(θ3),)))


### C5
α_list = [0.1:0.1:1;]


wlist1 = []
for i in α_list
  n = convex_weighting(zs1[1],zs2[1], i)
  append!(wlist1,n)
end

wlist2 = []
for i in α_list
  n = convex_weighting(zs2[1],zs3[1], i)
  append!(wlist2,n)
end

wlist3 = []
for i in α_list
  n = convex_weighting(zs3[1],zs1[1], i)
  append!(wlist3,n)
end


wlist1,wlist2,wlist3  = reshape(wlist1,2,10),reshape(wlist2,2,10),reshape(wlist3,2,10)

u10 = exp.(decoder2(wlist1))
u20 = exp.(decoder2(wlist2))
u30 = exp.(decoder2(wlist3))

p10 = u10 ./ (1 .+ u10)
p20 = u20 ./ (1 .+ u20)
p30 = u30 ./ (1 .+ u30)


image10 = mnist_img_alt(p10)
image20 = mnist_img_alt(p20)
image30 = mnist_img_alt(p30)

hcat(image10,image20,image30)

top = hcat(mnist_img_alt.([p10[:, i] for i in 1:10])...)
mid = hcat(mnist_img_alt.([p20[:, i] for i in 1:10])...)
bott = hcat(mnist_img_alt.([p30[:, i] for i in 1:10])...)

plot(vcat(top,mid,bott),
      xaxis=false, yaxis=false, titlefontsize=7, title = "10 Equally Spaced Alpha Interpolation - Bernoulli Means")
      plot!(size=(360,120))
savefig("Q3cf.pdf")

##### QUESTION 4 ####

### A.1

function top_half(x_flat)
  return (x_flat[1:392,:])
end


### A.2

function log_top(x,z)
  θ = top_half(decoder2(z))
  return sum(bernoulli_log_density(θ,top_half(x)), dims=1)
end

### A.3
function log_joint_top(x,z)
  return log_top(x,z) .+ log_prior(x)
end

# mnist_img_q4(x) = ndims(x)==2 ? Gray.(reshape(x,28,14,:)) : Gray.(transpose(reshape(x,28,14)))
# plot(mnist_img_q4(top_half(train_x[:,1])))


### B.1
n = 1
Random.seed!(414)
toy_mu = rand(2, n) # Initial mu, can initialize randomly!
toy_ls = rand(2, n)
params = [toy_mu, toy_ls]
###B.2
function elbo_top(params, x, num_samples)
    μ, logσ = params
    z = sample_diag_gaussian(μ,logσ)
    joint_ll = log_joint_top(x,z)
    log_q_z = log_q(μ, logσ, z)
    elbo_estimate = mean(joint_ll - log_q_z)
  return elbo_estimate
end

function top_loss(params, x; num_samples = 100)
  return -elbo_top(params, x, num_samples)
end


###B.3
# label of digits
label = train_data[2]
# create a table of the specific digit
train_one = train_x[:, findall(isequal(0),label)]
# select your image
train_one = train_one[:,1]

function fit_toy(init_params, x;  num_itrs=100, lr= 1e-2, num_q_samples =  50)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient((params) ->  top_loss(params, x; num_samples = num_q_samples), params_cur)[1]
    params_cur =  params_cur .- lr .*grad_params
    @info "loss $i: $(top_loss(params, x; num_samples = num_q_samples))"
  end
  return params_cur
end

trained_params_test = fit_toy(params,train_one; num_itrs = 200, lr= 1e-2, num_q_samples = 10000)

###B.4, contour function from A2
function skillcontour!(f; colour=nothing)
  n = 100
  x = range(-3,stop=-1,length=n)
  y = range(0,stop=2,length=n)
  z_grid = Iterators.product(x,y)
  z_grid = reshape.(collect.(z_grid),:,1)
  z = f.(z_grid)
  z = getindex.(z,1)'
  #max_z = maximum(z)
  levels = [.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2] .* maximum(z)
  if colour==nothing
  c = contour!(x, y, z, fill=false, levels=levels)
  else
  c = contour!(x, y, z, fill=false, c=colour,levels=levels,colorbar=false)
  end
  plot!(c)
end

z_1(zs) = exp.(log_joint_top(train_one, zs ))
z_2(zs) = exp.(factorized_gaussian_log_density(trained_params_test[1],trained_params_test[2],zs))
plot(xaxis = "z1", yaxis = "z2", title = "Approximate and True Posterior")
skillcontour!(z_1, colour=:red)
display(skillcontour!(z_2, colour=:blue))
#savefig("Q4bd.pdf")


###B.5
z = sample_diag_gaussian(trained_params_test[1], trained_params_test[2])

u40 = exp.(decoder2(z))
p40 = u40 ./(u40.+1)
mnist_img_alt(p30)
means_1= hcat(mnist_img_alt.([p40[:,i] for i = 1:1])...)

#original image
original = plot(mnist_img_alt(train_one))
#learnt image
mnist_img_alt(vec(p40))
#top half of original image with bottom of learnt image
top = top_half(train_one)
bottom = vec(p40)[393:end,:]
vec(vcat(top,bottom))
learnt = plot(mnist_img_alt(vec(vcat(top,bottom))))

plot(original, learnt , layout = (1,2), legend = false, xaxis = false, yaxis = false,
      title=["Original image" "Bottom half bernoulli means"])
#savefig("Q4be.pdf")
