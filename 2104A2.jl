Pkg.add("Revise")
Pkg.add("StatsFuns")
Pkg.add("MAT")
using Revise # lets you change A2funcs without restarting julia!
includet("A2_src.jl")
using Plots, Plots.PlotMeasures
using Statistics: mean
using Zygote
using Test
using Logging
using StatsFuns: log1pexp
#using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!
using Distributions: pdf, Normal
using Random

#prior over each player's skills is standard N distribution N(u,var)
# zs is KxN array, row is setting of skills for all N players

#### Q1 #####

### A #### joint prior
function log_prior(zs)
  prior = factorized_gaussian_log_density(0,0,zs)
  return prior
end

##### B ##### likelihood
function logp_a_beats_b(za,zb)
  x = zb-za
  return (-log1pexp.(x))
end

####### C ########
function all_games_log_likelihood(zs,games)
  m = logp_a_beats_b(zs[games[:,1],:], zs[games[:,2],:]) #index in to zs mat by games first col (mu) and 2nd col (sigma)
  return sum(m, dims=1) ##dims = 1 turns this output to a 1xK vector, k = skills
end

### [:,1] = first col
### first col of games are the winners, using that index for each winner, cycle through their skills in zs (NxK), do the same for the losers (2nd col of zs)

###### D ######## joint posterior
function joint_log_density(zs,games)
  return log_prior(zs) .+ all_games_log_likelihood(zs,games)
end

@testset "Test shapes of batches for likelihoods" begin
  B = 15 # number of elements in batch
  N = 4 # Total Number of Players
  test_zs = randn(4,15)
  test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
  @test size(test_zs) == (N,B)
  #batch of priors
  @test size(log_prior(test_zs)) == (1,B)
  # loglikelihood of p1 beat p2 for first sample in batch
  @test size(logp_a_beats_b(test_zs[1,1],test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
  @test size(logp_a_beats_b.(test_zs[1,:],test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
  @test size(all_games_log_likelihood(test_zs,test_games)) == (1,B) #B changed to N
  # batch loglikelihood under joint of evidence and prior
  @test size(joint_log_density(test_zs,test_games)) == (1,B)
end


###### Q2 #####

# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]',p1_wins), repeat([2,1]',p2_wins)]...)


# Example for how to use contour plotting code
plot(title="Log Prior Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill", margin=5mm
   )
example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.],[0.,0.5],zs))
#skillcontour!(example_gaussian; label="example gaussian")
skillcontour!(example_gaussian)
plot_line_equal_skill!()
savefig("Q2a.pdf")


#prior contour:
#a

f1(zs) = exp.(log_prior(zs))
t = display(plot(title="Joint Prior Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill", margin=5mm))
   display(skillcontour!(f1))
   display(plot_line_equal_skill!())
   savefig("2a.pdf")

#b
#
f11(zs) = exp.(logp_a_beats_b(zs[1,:],zs[2,:]))
t = display(plot(title="Log A beats B",
        xlabel = "Player 1 Skill",
        ylabel = "Player 2 Skill", margin=5mm))
         #prior(zs) = exp.(log_prior(zs))
        display(skillcontour!(f11))
        display(plot_line_equal_skill!())
         # savefig("2d.pdf")
         savefig("2b.pdf")

##### c #######
a_win_1 = two_player_toy_games(1, 0)
a_win_10 = two_player_toy_games(10, 0)
split_match_10 = two_player_toy_games(10, 10)


f2(zs) = exp(joint_log_density(zs,a_win_1))
t = display(plot(title="Joint Posterior Contour Plot (1 Game)",
        xlabel = "Player 1 Skill",
        ylabel = "Player 2 Skill", margin=5mm))
         #prior(zs) = exp.(log_prior(zs))
        display(skillcontour!(f2))
        display(plot_line_equal_skill!())
         # savefig("2d.pdf")
        savefig("2c.pdf")

####### d ########
f3(zs) = exp(joint_log_density(zs,a_win_10))
t = display(plot(title="Joint Posterior Contour Plot (10 Games)",
       xlabel = "Player 1 Skill",
       ylabel = "Player 2 Skill", margin=3mm))
        #prior(zs) = exp.(log_prior(zs))
       display(skillcontour!(f3))
       display(plot_line_equal_skill!())
       savefig("2d.pdf")

########## e ############
f4(zs) = exp(joint_log_density(zs,split_match_10))
t = display(plot(title="Joint Posterior Contour Plot (20 Games)",
       xlabel = "Player 1 Skill",
       ylabel = "Player 2 Skill", margin=3mm))
        #prior(zs) = exp.(log_prior(zs))
       display(skillcontour!(f4))
       display(plot_line_equal_skill!())
       savefig("2e.pdf")


############ Q3 a ############

function elbo(params,logp,num_samples)
  x = randn(size(params[1])[1],num_samples)
  a = exp.(params[2])
  b = params[1]
  samples = a .* x .+ b
  logp_estimate = logp(samples)
  logq_estimate = factorized_gaussian_log_density(params[1],params[2],samples)
  return sum(logp_estimate - logq_estimate)/(num_samples)
end

############ b ################

# para = [[1 2]', [4 5]]
# para[1]
# size(para[1])[1]

# Conveinence function for taking gradients
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  logp(zs) = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end

############ c ############
# Toy game
num_players_toy = 2
toy_mu = [-2.,3.] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.] # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)

function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient((params) -> neg_toy_elbo(params; games = toy_evidence, num_samples = num_q_samples), params_cur)[1]
    params_cur =  params_cur .- lr .* grad_params
    @info  "Loss $(neg_toy_elbo(params_cur; games = toy_evidence, num_samples = num_q_samples))"

    plot();

    target(zs) = exp.(joint_log_density(zs, toy_evidence))
    skillcontour!(target, colour=:red)
    plot_line_equal_skill!()
    variational(zs) = exp.(factorized_gaussian_log_density(params_cur[1],params_cur[2],zs))
    display(skillcontour!(variational, colour=:blue))
  end
  return params_cur
end


####### d #######
fit_toy_variational_dist(toy_params_init, a_win_1)
plot!(title="fit q with SVI observing player A winning 1 game",
        xlabel = "Player 1 Skill",
        ylabel = "Player 2 Skill", margin=3mm)
        savefig("3d.pdf")

####### e ######
fit_toy_variational_dist(toy_params_init, a_win_10)
plot!(title="fit q with SVI observing player A winning 10 games",
       xlabel = "Player 1 Skill",
       ylabel = "Player 2 Skill", margin=3mm)
       savefig("3e.pdf")

######### f ######
fit_toy_variational_dist(toy_params_init, split_match_10 )
plot!(title="fit q with SVI both winning 10 games",
       xlabel = "Player 1 Skill",
       ylabel = "Player 2 Skill", margin=3mm)
       savefig("3f.pdf")

cd("/Users/justinleo/Documents/GitHub/STA414-2020-A2-leojusti/")

## Question 4
# Load the Data
using MAT
vars = matread("tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")

##### b ##########
function fit_variational_dist(init_params, tennis_games; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient((params) -> neg_toy_elbo(params; games = tennis_games, num_samples = num_q_samples), params_cur)[1]
    params_cur =  params_cur .- lr .* grad_params
    @info  "Loss $(neg_toy_elbo(params_cur; games = tennis_games, num_samples = num_q_samples))"
  end
  return params_cur
end

#### c ############
init_mu = randn(num_players)
init_log_sigma = rand(num_players)
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games)
means = trained_params[1]
logstd = trained_params[2]

# return players index ordered worst to best
perm = sortperm(means)

# plots player ranking vs player 2 skill, 107th player is top player with highest mean
plot()
plot(means[perm],  yerror = exp.(logstd[perm]), title="Approx. Mean & Var of all players",
       xlabel = "Player Ranking",
       ylabel = "Player Skill", label = "Approx. Mean", margin=5mm)
       savefig("4c.pdf")

### e ##########

ordered_list = sortperm(means, rev=true)
top_ten_players = player_names[ordered_list][1:10]
fed_i = findall(x->x=="Roger-Federer", player_names)[1][1]
nad_i = findall(x->x=="Rafael-Nadal", player_names)[1][1]
fed_nad_index = [fed_i,nad_i]
skill_m = trained_params[1][fed_nad_index]

#using trained para defined earlier, which returns list of mu and sigmas, index into it and get mus,
# then index into the mus for nadal and federer

games_w_fed = findall(x->x==fed_i, tennis_games)
games_w_nad = findall(x->x==nad_i, tennis_games)

nlist = []
games = 0
for i in games_w_fed
  games = i[1]
  append!(nlist,games)
end

## gamEs where fed played = nlist
#nlist

fed = tennis_games[nlist,:]
fed_nad_games = findall(x->x==1,fed)

n2list = []
for i in fed_nad_games
  games2 = i[1]
  append!(n2list,games2)
end

FN = fed[n2list,:]
for i in 1:8
  if FN[i] == 5
    FN[i] = 2
  end
end

FN_plot(zs) = exp.(joint_log_density(zs,FN))
plot( title="Federer & Nadal Plot",
       xlabel = "Federer Skill",
       ylabel = "Nadal Skill", margin=5mm)
       display(skillcontour!(FN_plot))

plot_line_equal_skill!()
savefig("4e.pdf")

######## f ###########


#### line of best fit becomes flatter with the transformation
### likelihood = conditional probability


#### g ######

raf_mu = means[nad_i]
fed_mu = means[fed_i]

raf_sig = exp(logstd[nad_i])
fed_sig = exp(logstd[fed_i])

ya_mu = fed_mu - raf_mu
ya_sig = fed_sig^2 + raf_sig^2

using Distributions: cdf

prob_Fed = 1 - cdf(Normal(ya_mu, ya_sig), 0)

function monte_carlo(mu1, sigma1, mu2, sigma2, iters)
  num = 0
  a = 0
  b = 0
  for i in 1:iters
    a = sigma1 .* randn(1) .+ mu1
    b = sigma2 .* randn(1) .+ mu2
    #println(a,b)
    if a > b
      num += 1
    end
  end
  return num/iters
end


monte_carlo(fed_mu, fed_sig, raf_mu, raf_sig, 10000)


####### h #####

lowest_mu = means[ordered_list[end]]
lowest_sig = exp(logstd[ordered_list[end]])

ya_mu2 = fed_mu - lowest_mu
ya_sig2 = exp(fed_sig)

prob_Fed2 = 1 - cdf(Normal(ya_mu2, ya_sig2), 0)

monte_carlo(fed_mu, fed_sig, lowest_mu, lowest_sig, 10000)
