using Agents, Random, DataFrames, LightGraphs
using Distributions
using DrWatson: @dict
using Plots
using Random
using InteractiveDynamics
using CairoMakie
using BlackBoxOptim, Random
using Statistics: mean
using Gadfly
using LinearAlgebra

#how many students
num_students = 763

#Agent type
mutable struct Student <: AbstractAgent
    id::Int
    status::Symbol  # 1: S, 2: I, 3:R
end

function model_initialize(; num_students = 763, beta = 2, gamma = 0.5, seed = 500)
    rng = Random.MersenneTwister(seed)
    # create properties for gamma and beta which can then be changed later
    # seed passed in as a property due to errors
    properties = Dict(:beta => beta, :gamma => gamma, :seed => seed)
    #student agents with a random number generator (not working)
    model = ABM(Student; properties)

     #add agents
    for i in 1:num_students
        add_agent!(model, :S)
    end

    ##infect a student
    agent = model[1]
    agent.status = :I

    return model
end



##the model step function
function model_step!(model)
    transmit_and_recover!(model)
end

##functions to calculate the infected, recovered and Susceptible population
total_infected(m) = count(a.status == :I for a in allagents(m))
total_recovered(m) = count(a.status == :R for a in allagents(m))
total_sus(m) = count(a.status == :S for a in allagents(m))

##transmit and recover function for the model step
function transmit_and_recover!(model)
    seed = model.seed
    rng = Random.MersenneTwister(seed)

    ##caculate current values
    total_inf = total_infected(model)
    total_rec = total_recovered(model)
    total_suspec = total_sus(model)

    ##read beta value from model property
    beta_param = model.beta
    index = 0

    ##returns int value of number infected
    new_infected = rand(rng,Poisson(total_suspec * total_inf * beta_param / num_students))

    ##infect students based on how above formula
    for a in allagents(model)
        if (index < new_infected)
            if a.status == :S
                a.status = :I
                index += 1
            end
        end
    end



##recover portion

    Y = model.gamma
    counter = 0

    new_recover = rand(rng, Poisson(total_inf*Y))

    cur_infected = total_infected(model)

    valid_recover = min(new_recover,cur_infected)

    for a in allagents(model)
        if (counter < valid_recover)
            if a.status == :I
                a.status = :R
                counter += 1
            end
        end
    end

end

## data collection and run model

#create the model
# model = model_initialize()
#
# # #models with different seeds
# model_1 = model_initialize(seed = 125)
# model_2 = model_initialize(seed = 10)
# model_3 = model_initialize(seed = 500)
#
#
# adata = [:status]
#
# mdata = [total_infected, total_recovered, total_sus]
#
# _, model_df_1 = run!(model_1,dummystep,model_step!,10; adata, mdata)
# _, model_df_2 = run!(model_2,dummystep,model_step!,10; adata, mdata)
# _, model_df_3 = run!(model_3,dummystep,model_step!,10; adata, mdata)


  #parameters = Dict(:beta => collect(2:3), :gamma => [0.5,0.6],)

# parameters = Dict(:seed => [10,500])
# # #
#   _, model_df = paramscan(parameters, model_initialize; adata, mdata, model_step!, n = 10)



## Optimising function

#actual values
# actuals = [3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]



#model_non_op = model_initialize(beta = 2, gamma = 0.5)
model_op = model_initialize(beta = x[1], gamma = x[2])

adata = [:status]

mdata = [total_infected, total_recovered, total_sus]

#_, model_df_1 = run!(model_non_op,dummystep,model_step!,10; adata, mdata)
_, model_df_2 = run!(model_op,dummystep,model_step!,10; adata, mdata);

#plot the one with optimal params by eye
model_1 = model_initialize(beta = 2.1, gamma = 0.85, seed = 50)

_, model_df_1 = run!(model_1,dummystep,model_step!,10; adata, mdata)



## one time step function

function observe(v)
    return (v[2])

end

# DJS to change this to accept HA's function which takes S I R beta
# gamma and returns S I R one time step ahead.
function simulate_one_step(vec)
    seed = 500
    rng = Random.MersenneTwister(seed)

    ##values of SIR that were passed in as params
    oldS  = vec[1]
    oldI  = vec[2]
    oldR  = vec[3]
    beta  = vec[4]
    gamma = vec[5]

    ##returns int value of number infected
    new_infected = rand(rng, Poisson(oldS * oldI * beta / num_students))
    valid_infected = min(new_infected, oldS)

    #determine the number of newly recovered individuals
    new_recover = rand(rng, Poisson(oldI * gamma))
    valid_recover = min(new_recover, oldI)

    newS = oldS - valid_infected
    newI = oldI + valid_infected - valid_recover
    newR = oldR + valid_recover

    output = [newS newI newR]

    return (output)

end

# ------------------
# Pariticle Filtering
# ------------------

# Parameter update covariace aka parameter diffusivity
Q = [0.1 0.0; 0.0 0.01];
# Observation covariance
R = Matrix{Float64}(undef,1,1);
R[1,1] = 0.1;
# Number of particles
N = 1000;
# S, I, R beta and gamma
nx = 5;
# S and I since S + I + R = 763 always - no boys die
ny = 3;

function resample_stratified(weights)

    N = length(weights)
    # make N subdivisions, and chose a random position within each one
    positions =  (rand(N) + collect(range(0, N - 1, length = N))) / N

    indexes = zeros(Int64, N)
    cumulative_sum = cumsum(weights)
    i, j = 1, 1
    while i <= N
        if positions[i] < cumulative_sum[j]
            indexes[i] = j
            i += 1
        else
            j += 1
        end
    end
    return indexes
end




function pf(inits, N, f, hh, y, Q, R, nx, ny)
    # inits - initial values

    # N - number of particles

    # f is the state update function - for us this is the ABM - takes
    # a state and the parameters and returns a new state one time step
    # forward

    # h is the observation function - for us the state is S, I, R
    # (although S + I + R = total boys) but we can only observe I

    # y is the data - for us this the number infected for each of the
    # 14 days

    # To avoid particle collapse we need to perturb the parameters -
    # we do this by sampling from a multivariate normal distribution
    # with a given covariance matrix Q

    # R is the observation covariance matrix

    # nx is the dimension of the state - we have S I R beta and gamma
    # but S + I + R is fixed so we can take the state to be S I beta
    # gamma i.e. nx = 4

    # The state update function (running the ABM) returns the new
    # state so we need to know which of the nx state variables are the
    # actual state and which are parameters which we now consider to
    # be state. ny is the number of actual state variables so that
    # nx - ny is the number of parameters.

    T = length(y)
    log_w = zeros(T,N);
    x_pf = zeros(nx,N,T);
    x_pf[:,:,1] = inits;
    wn = zeros(N);

    for t = 1:T
        if t >= 2
            a = resample_stratified(wn);
            #f is a vector S I R beta gamma - vector of len 5
            x_pf[1 : ny, :, t] = hcat(map(f, mapslices(z -> [z], x_pf[:, a, t - 1], dims = 1))...)
            log_x_pf = map(log, x_pf[ny + 1 : nx, a, t - 1])
            epsilons = rand(MvNormal(zeros(nx - ny), Q), N)
            new_log_x_pf = log_x_pf .+ epsilons
            x_pf[ny + 1 : nx, :, t] = map(exp, new_log_x_pf)
        end

        log_w[t, :] = logpdf(MvNormal(y[t, :], R), map(observe, mapslices(z -> [z], x_pf[:,:,t], dims=1)))

        # To avoid underflow subtract the maximum before moving from
        # log space

        wn = map(x -> exp(x), log_w[t, :] .- maximum(log_w[t, :]));
        wn = wn / sum(wn);
    end

    log_W = sum(map(log, map(x -> x / N, sum(map(exp, log_w[1:T, :]), dims=2))));

    return(x_pf, log_w, log_W)

end

# nx = S, I, R and beta gamma
inits = zeros(nx, N)
inits[1, :] .= 762;
inits[2, :] .= 1;
inits[3, :] .= 0;
inits[4, :] .= rand(LogNormal(log(2),0.01),N)
inits[5, :] .= rand(LogNormal(log(0.5),0.005),N);


#actual values
y = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]

(foo, bar, baz) = pf(inits, N, simulate_one_step, observe, y, Q, R, nx, ny);






## Plotting function
#optimised version
#Plots.plot!(model_df_2.step, actuals[1:11], title = "Total population infected", labels = "Actual version", legend = :left, xlabel = "Day", ylabel = "Number of cases")

Gadfly.plot(layer(x = model_df_2.step, y = model_df_2.total_infected, Geom.line, Gadfly.Theme(default_color=color("red"))),
            layer(x = model_df_2.step, y = actuals[1:11], Geom.line, Gadfly.Theme(default_color=color("blue"))),
            layer(x = model_df_1.step, y = model_df_1.total_infected, Geom.line, Gadfly.Theme(default_color=color("green"))),
            Guide.XLabel("Day"),
            Guide.YLabel("Population"),
            Guide.Title("Actual flu cases against different simulation versions"),
            Guide.manual_color_key("Legend", ["Optimised version", "Actual data", "Approximated version"], ["red", "blue", "green"]))


Gadfly.plot(layer(x = model_df_2.step, y = model_df_2.total_recovered, Geom.line, Gadfly.Theme(default_color=color("red"))),
            layer(x = model_df_1.step, y = model_df_1.total_recovered, Geom.line, Gadfly.Theme(default_color=color("green"))),
            Guide.XLabel("Day"),
            Guide.YLabel("Population"),
            Guide.Title("Recovered lu cases simulated with different parameters"),
            Guide.manual_color_key("Legend", ["Optimised version","Approximated version"], ["red", "green"]))


Gadfly.plot(layer(x = model_df_2.step, y = model_df_2.total_sus, Geom.line, Gadfly.Theme(default_color=color("red"))),
            layer(x = model_df_1.step, y = model_df_1.total_sus, Geom.line, Gadfly.Theme(default_color=color("green"))),
            Guide.XLabel("Day"),
            Guide.YLabel("Population"),
            Guide.Title("Susceptible simulated with different parameters"),
            Guide.manual_color_key("Legend", ["Optimised version","Approximated version"], ["red", "green"]))
