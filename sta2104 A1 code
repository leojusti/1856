using Distributions
using StatsPlots
using Test, Plots, Plots.PlotMeasures

#Q1

#1.1

losses = [[10, 0], [1, 50],[0, 200]]

num_actions = length(losses)

Show = losses[1]
Folder = losses[2]
Delete = losses[3]

function expected_loss_of_action(p_spam, action)
#TODO: Return expected loss over a Bernoulli random variable
# with mean prob spam.
# Losses are given by the table above.
    if action == 1
        time = Show
    elseif action == 2
        time = Folder
    else
        time = Delete

    end
#prob_spam = 11/261
    #p = rand(Bernoulli(mean(p_spam)))

    list = []
    for i in p_spam
        value = i.*(time[1]) .+ (1-i).*(time[2])
        #print(value)
        append!(list,value)
    end


    return list
end

prob_range = range(0., stop=1., length=500)



p1 = collect(prob_range)

# Make plot


plot(xaxis = ("Probability"), yaxis = ("Expected Loss"))
plot()
using Plots
for action in 1:num_actions
    names = ("Show", "Folder", "Delete")
    display(plot!(p1, expected_loss_of_action((p1), action),label = (names[action]), xaxis = ("Probability"), yaxis = ("Expected Loss")))
    #display(plot!(p1, expected_loss_of_action(p1, action))
end

savefig("A1plot")

#1.2

function optimal_action(prob_spam)
  list = []
  for i in 1:num_actions
    value = expected_loss_of_action(prob_spam,i)
    append!(list, value)
  end
  min = findmin(list)[2]
  #actions =  ["Show","Folder","Delete"]
  return min
end

#1.3

prob_range = range(0, stop=1., length=500)
optimal_losses = []
optimal_actions = []
for p in prob_range
    v = optimal_action(p)
    append!(optimal_actions,v)
    append!(optimal_losses,expected_loss_of_action(p,v))
# TODO: Compute the optimal action and its expected loss for
# probability of spam given by p.
end
plot(prob_range, optimal_losses, linecolor=optimal_actions, title = "Optimal Action wrt Probability", xlabel = "x", ylabel = "y")

savefig("A1plot2")




############


#Q3

using LinearAlgebra
################################## Regression

## Toy Data [2pts]

# 1
function target_f1(x, σ_true=0.3)
  noise = randn(size(x))
  y = 2x .+ σ_true.*noise
  return vec(y)
end

function target_f2(x)
  noise = randn(size(x))
  y = 2x + norm.(x)*0.3.*noise
  return vec(y)
end

function target_f3(x)
  noise = randn(size(x))
  y = 2x + 5sin.(0.5*x) + norm.(x)*0.3.*noise
  return vec(y)
end

# 1
function sample_batch(target_f, batch_size)
  x = rand(Uniform(0,20),batch_size)'
  y = target_f(x)
  return x,y
end

@testset "sample dimensions are correct" begin
  m = 1 # dimensionality
  n = 200 # batch-size
  for target_f in (target_f1, target_f2, target_f3)
    x,y = sample_batch(target_f,n)
    @test size(x) == (m,n)
    @test size(y) == (n,)
  end
end

# 2
n=1000
x1,y1 = sample_batch(target_f1,n)
  plot_f1 = scatter(x1',y1,title = "Target f1", xlabel = "x", ylabel = "y",label = "Data")
savefig("A1q2.1")
x2,y2 =sample_batch(target_f2,n)
  plot_f2 = scatter(x2',y2,title = "Target f2", xlabel = "x", ylabel = "y",label = "Data")
savefig("A1q2.2")
x3,y3 = sample_batch(target_f3,n)
  plot_f3 = scatter(x3',y3,title = "Target f3", xlabel = "x", ylabel = "y",label = "Data")
savefig("A1q2.3")
## Linear Regression Model with $\hat \beta$ MLE [4pts]
# 1. [2pts] Program the function that computes the the maximum likelihood estimate given $X$ and $Y$.
function beta_mle(x,y)
  beta = inv(x*x')*(x*y)
  return beta
end

n=1000


x1,y1 = sample_batch(target_f1,n)

x2,y2 =sample_batch(target_f2,n)

x3,y3 = sample_batch(target_f3,n)


lobf_mle_1(x) = x'.*beta_mle(x1,y1)
lobf_mle_2(x) = x'.*beta_mle(x2,y2)
lobf_mle_3(x) = x'.*beta_mle(x3,y3)


# 2. [2pts] For each function, plot the linear regression model given by $Y \sim \mathcal{N}(X^T\hat\beta, \sigma^2 I)$ for $\sigma=1.$.

plot_f1_v2 = scatter(x1',y1)
  display(plot!(plot_f1_v2, lobf_mle_1, ribbon= 1.,title = "Linear Regression Model - Target f1", xlabel = "x", ylabel = "y",label = "Fitted Line + Ribbon"))
savefig("A1q2.4")
plot_f2_v2 = scatter(x2',y2)
  display(plot!(plot_f2_v2, lobf_mle_2, ribbon= 1.,title = "Linear Regression Model - Target f2", xlabel = "x", ylabel = "y",label = "Fitted Line + Ribbon"))
savefig("A1q2.5")
plot_f3_v2 = scatter(x3',y3)
  display(plot!(plot_f3_v2, lobf_mle_3, ribbon= 1.,title = "Linear Regression Model - Target f3", xlabel = "x", ylabel = "y",label = "Fitted Line + Ribbon"))
savefig("A1q2.6")

## Log-likelihood of Data Under Model [6pts]

# 1. [2pts] Write code for the function that computes the likelihood of $x$ under the Gaussian distribution $\mathcal{N}(μ,σ)$.
function gaussian_log_likelihood(μ,σ,x)
  n = length(x)
  return (-n/2)*log(2*pi*σ^2)-(1/(2*(σ^2)))*sum((x-μ)^2)
end

# Test Gaussian likelihood against standard implementation
@testset "Gaussian log likelihood" begin
  using Random #TODO: added to fix -inf problem
  Random.seed!(123)
  # Scalar mean and variance
  x = randn()
  μ = randn()
  σ = rand()
  @test size(gaussian_log_likelihood(μ,σ,x)) == () # Scalar log-likelihood
  @test gaussian_log_likelihood.(μ,σ,x) ≈ log.(pdf.(Normal(μ,σ),x))
   # Correct Value
  # Vector valued x under constant mean and variance
  x = randn(100)
  μ = randn()
  σ = rand()
  @test size(gaussian_log_likelihood.(μ,σ,x)) == (100,) # Vector of log-likelihoods
  @test gaussian_log_likelihood.(μ,σ,x) ≈ log.(pdf.(Normal(μ,σ),x)) # Correct Values
  # Vector valued x under vector valued mean and variance
  x = randn(10)
  μ = randn(10)
  σ = rand(10)
  @test size(gaussian_log_likelihood.(μ,σ,x)) == (10,) # Vector of log-likelihoods
  @test gaussian_log_likelihood.(μ,σ,x) ≈ log.(pdf.(Normal.(μ,σ),x)) # Correct Values
end


# 2. [2pts] Use your gaussian log-likelihood function to write the code which computes the negative log-likelihood of the target value $Y$ under the model $Y \sim \mathcal{N}(X^T\beta, \sigma^2*I)$ for a given value of $\beta$.
function lr_model_nll(β,x,y;σ=1.)
  return sum((-1)*(gaussian_log_likelihood.(x'*β, σ, y)))
end

# 3. [1pts] Use this function to compute and report the negative-log-likelihood of a $n\in \{10,100,1000\}$ batch of data
for n in (10,100,1000)
    println("--------  $n  ------------")
    for target_f in (target_f1,target_f2, target_f3)
      println("--------  $target_f  ------------")
      nll_vec = []
      for σ_model in (0.1,0.3,1.,2.)
        println("--------  $σ_model  ------------")
        x,y = sample_batch(target_f,n)
        β_mle = beta_mle(x,y)[1]
        nll = lr_model_nll(β_mle,x,y; σ=σ_model)
        println("Negative Log-Likelihood: $nll")
        append!(nll_vec,nll)
      end
    nLog =   findmin(nll_vec)
    op_nll = nLog[1]
    op_sig = (0.1,0.3,1.,2.)[nLog[2]]
    println("Min. Negative Log Likelihood: $op_nll " , " Optimal Signal: $op_sig") #$binds variable from function to string
    end
end

# 4. [1pts] For each target function, what is the best choice of $\sigma$?

## Automatic Differentiation and Maximizing Likelihood [3pts]

# 1.
@testset "Gradients wrt parameter" begin
  β_test = randn()
  σ_test = rand()
  x,y = sample_batch(target_f1,100)
  ad_grad = gradient((β) -> lr_model_nll(β,x,y;σ=σ_test), β_test)
  #hand_derivative = -(sum((y .- (x'.*β_test)).*x') / (σ_test^2))
  hand_derivative = -(x *y - x * x'*β_test) / (σ_test^2) #TODO: the new correct hand_derivative based on x in (mxn) not the old x in (nxm)
  @test ad_grad[1] ≈ hand_derivative
end

### Train Linear Regression Model with Gradient Descent [5pts]

# 1. [3pts] Write a function `train_lin_reg` that accepts a target function and an initial estimate for $\beta$ and some
using Logging # Print training progress to REPL, not pdf
function train_lin_reg(target_f, β_init; bs= 100, lr = 1e-6, iters=1000, σ_model = 1. )
  β_curr = β_init
  #grad_β = 0 TODO: Remove this line
  for i in 1:iters
    x,y = sample_batch(target_f,bs)
    @info "loss: $(lr_model_nll(β_curr,x,y;σ=σ_model)) β: $β_curr"
    grad_β = gradient((β) -> lr_model_nll(β,x,y;σ=σ_model), β_curr)[1]
    #β_curr = β_curr + grad_β
    β_curr -= (grad_β) * lr
  end
  return β_curr
end


# 2. [2pts] For each target function, start with an initial parameter $\beta$,
  #  learn an estimate for $\beta_\text{learned}$ by gradient descent.
  #  Then plot a $n=1000$ sample of the data and the learned linear regression model with shaded region for uncertainty corresponding to plus/minus one standard deviation.

β_init = 1000*randn()
β1_learned = train_lin_reg(target_f1,β_init)
β2_learned= train_lin_reg(target_f2,β_init)
β3_learned= train_lin_reg(target_f3,β_init)

v1(x) = x'.*β1_learned
v2(x) = x'.*β2_learned
v3(x) = x'.*β3_learned

plot_f1_new = scatter(x1',y1, label = "Data")
  display(plot!(plot_f1_new, v1, ribbon=1., title = "Trained Linear Regression Model - Target f1", xlabel = "x", ylabel = "y",label = "Fitted Line"))
savefig("A1q2.7")

plot_f2_new = scatter(x2',y2,label = "Data")
  display(plot!(plot_f2_new, v2, ribbon=1.,title = "Trained Linear Regression Model - Target f2", xlabel = "x", ylabel = "y",label = "Fitted Line"))
savefig("A1q2.8")

plot_f3_new = scatter(x3',y3,label = "Data")
  display(plot!(plot_f3_new, v3, ribbon=1.,title = "Trained Linear Regression Model - Target f3", xlabel = "x", ylabel = "y",label = "Fitted Line"))
savefig("A1q2.9")
### Non-linear Regression with a Neural Network [9pts]
# 1. [3pts] Write the code for a fully-connected neural network (multi-layer perceptron) with one 10-dimensional hidden layer and a `tanh` nonlinearirty.
# Neural Network Function
function neural_net(x,θ)
  w1 = θ[1]
  b1 = θ[2]
  w2 = θ[3]
  b2 = θ[4]
  return tanh.(x'*w1 .+ b1)*w2 .+ b2
end

θ = (randn(1, 10), randn(1,10), randn(10,), randn(1,))

d = randn(1,6)

d'.*θ[1] .+ θ[2]

neural_net(d,θ)

@testset "neural net mean vector output" begin
  n = 100
  x,y = sample_batch(target_f1,n)
  µ = neural_net(x,θ)
  @test size(µ) == (n,)
end

# 2. [2pts] Write the code that computes the negative log-likelihood for this model where the mean is given by the output of the neural network and $\sigma = 1.0$
function nn_model_nll(θ,x,y;σ=1)
  n_Net = neural_net(x,θ)
  return sum(-1*gaussian_log_likelihood.(n_Net, σ, y))
end

using Logging # Print training progress to REPL, not pdf
function train_nn_reg(target_f, θ_init; bs= 100, lr = 1e-5, iters=100, σ_model = 1. )
    θ_curr = θ_init
    for i in 1:iters
      x,y = sample_batch(target_f,bs)
      @info "loss: $(nn_model_nll(θ_curr,x,y;σ=σ_model))" # θ: $θ_curr" -- this provides no useful info and it just wasteful
      grad_θ = gradient((θ) -> nn_model_nll(θ,x,y;σ=1), θ_curr)[1]
      θ_curr = θ_curr .- lr .*grad_θ
    end
    return θ_curr
end

#### plots

θ_init = θ
θ1_learned =  train_nn_reg(target_f1, θ_init; bs=1000, iters=7000, lr=1e-5)
θ2_learned =  train_nn_reg(target_f2, θ_init; bs=1000, iters=10000, lr=1e-6)
θ3_learned =  train_nn_reg(target_f3, θ_init; bs=1000, iters=10000, lr=1e-6)



x1_new,y1_new = sample_batch(target_f1, 1000)
plot_f1_new = scatter(x1_new', y1_new,label = "Data")
  display(plot!(plot_f1_new, sort(x1_new'),sort(neural_net(x1_new, θ1_learned)), ribbon=1., title = "Non Linear Regression with Neural Network - Target f1", xlabel = "x", ylabel = "y", label = "Fitted Line"))
savefig("A1q2.10")

x2_new,y2_new = sample_batch(target_f2, 1000)
plot_f2_new = scatter(x2_new', y2_new,label = "Data")
  display(plot!(plot_f2_new, sort(x2_new'),sort(neural_net(x2_new, θ2_learned)), ribbon=1., title = "Non Linear Regression with Neural Network - Target f2", xlabel = "x", ylabel = "y",label = "Fitted Line"))
savefig("A1q2.11")

x3_new,y3_new = sample_batch(target_f3, 1000)
plot_f3_new = scatter(x3_new', y3_new,label = "Data")
  display(plot!(plot_f3_new, sort(x3_new'),sort(neural_net(x3_new, θ3_learned)), ribbon=1.,title = "Non Linear Regression with Neural Network - Target f3", xlabel = "x", ylabel = "y",label = "Fitted Line"))
savefig("A1q2.12")




### Non-linear Regression and Input-dependent Variance with a Neural Network [8pts]
# 1. [1pts]
# Neural Network Function
function neural_net_w_var(x,θ)
  w1 = θ[1]
  b1 = θ[2]
  w2 = θ[3]
  b2 = θ[4]
  m = tanh.(x'*w1 .+ b1)*w2 .+ b2
  μ = m[:,1]
  logσ = m[:,2]
  return μ, logσ
end

# Random initial Parameters
Random.seed!(4)
θ_new = (randn(1, 10), randn(1,10), randn(10,2), randn(1,2))

@testset "neural net mean and logsigma vector output" begin
  n = 100
  x,y = sample_batch(target_f1,n)
  μ, logσ = neural_net_w_var(x,θ_new)
  @test size(μ) == (n,)
  @test size(logσ) == (n,)
end

# 2. [2pts]
function nn_with_var_model_nll(θ,x,y)
  μ,σ = neural_net_w_var(x,θ)
  return sum(-1*gaussian_log_likelihood.(μ, exp.(σ), y))
end

# 3. [1pts]
function train_nn_w_var_reg(target_f, θ_init; bs= 1000, lr = 1e-4, iters=10000)
    θ_curr = θ_init
    for i in 1:iters
      x,y = sample_batch(target_f,bs)
      @info "loss: $(nn_with_var_model_nll(θ_curr,x,y))"
      grad_θ = gradient((θ) -> nn_with_var_model_nll(θ,x,y), θ_curr)[1]
      θ_curr = θ_curr .- lr .*grad_θ
    end
    return θ_curr
end

# 4. [4pts]
Random.seed!(4)
θ_init = θ_new
θ1_new_learned =  train_nn_w_var_reg(target_f1, θ_init; bs=500, lr = 1e-5, iters=15000)
θ2_new_learned =  train_nn_w_var_reg(target_f2, θ_init; lr = 1e-5, iters=15000)
θ3_new_learned =  train_nn_w_var_reg(target_f3, θ_init; lr = 1e-5, iters=20000)



x1_new2,y1_new2 = sample_batch(target_f1, 1000)
plot_f1_new2 = scatter(x1_new2', y1_new2, label="Data"
  ,xlabel="x",ylabel="y",margin=5mm, title="NLR with Fitted Line and Variance - Target1 ")
  display(plot!(plot_f1_new2, vec(x1_new2), neural_net_w_var(x1_new2, θ1_new_learned)
    , label="Neural Network Fitted Line", seriestype=:line
    , ribbon =neural_net_w_var(x1_new2, θ1_new_learned)[2]))
      savefig("lobf_nn__wvar_learned_1.pdf")

x2_new2,y2_new2 = sample_batch(target_f2, 1000)
plot_f2_new2 = scatter(x2_new2', y2_new2, legend=:topleft, label="Data"
  ,xlabel="x",ylabel="y",margin=5mm, title="NLR with Fitted Line and Variance - Target2 ")
  display(plot!(plot_f2_new2, vec(x2_new2), neural_net_w_var(x2_new2, θ2_new_learned)[1]
    , label="Neural Network Fitted Line",seriestype=:line
    , ribbon =neural_net_w_var(x2_new2, θ2_new_learned)[2]))
      savefig("lobf_nn__wvar_learned_2.pdf")

x3_new2,y3_new2 = sample_batch(target_f3, 1000)
plot_f3_new2 = scatter(x3_new2', y3_new2, legend=:topleft, label="Data"
  ,xlabel="x",ylabel="y",margin=5mm, title="NLR with Fitted Line and Variance - Target3")
  display(plot!(plot_f3_new2, vec(x3_new2), neural_net_w_var(x3_new2, θ3_new_learned)[1]
    , label="Neural Network Fitted Line",seriestype=:line
    , ribbon =neural_net_w_var(x3_new2, θ3_new_learned)[2]))
      savefig("lobf_nn__wvar_learned_3.pdf")
