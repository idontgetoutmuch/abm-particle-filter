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


## one time step function

#to observe the infected
function observe(v)
    return (v[2])

end

seed = 500
rng = Random.MersenneTwister(seed)


#Simulates one time step and returns the S, I , R, Beta and Gamma
function simulate_one_step(vec)
    #seed = 500
    #rng = Random.MersenneTwister(seed)

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

    #determine the new number of susceptible, infected and recovered
    newS = oldS - valid_infected
    newI = oldI + valid_infected - valid_recover
    newR = oldR + valid_recover

    #in a vector format
    output = [newS newI newR]

    return (output)

end

# ------------------
# Particle Filtering
# ------------------

# Parameter update covariace aka parameter diffusivity
Q = [0.1 0.0; 0.0 0.01]; #--> spread is ok, peak is not
#Q = [0.01 0.0; 0.0 0.01]; #--> spread does not change, peak is flattened
#Q = [0.001 0.0; 0.0 0.001]; #--> spread effected, peak is flattened
#Q = [0.005 0.0; 0.0 0.05]; #--> does not reflect spread

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


# foo is a 5x1000x15 array --> so observe 5 variables, 1000 particles and 15 time steps
# so {:,:,1} = 762 762 = mean(row)
#              1.0 1.0 = mean(row) --> these means will then be the point for t = 1
(foo, bar, baz) = pf(inits, N, simulate_one_step, observe, y, Q, R, nx, ny);

## Plots of mean sus, infected, recovered, beta and gamma and std for these

overall_mean_matrix = zeros(15,5)
overall_std_matrix = zeros(15,5)
#calculate mean based on time step and store in a overall matrix for plotting
for i in 1:15
    (INF, REC,SUS,BE,GA) = mean(foo[:,:,i], dims = 2)
    overall_mean_matrix[i,1] = INF
    overall_mean_matrix[i,2] = REC
    overall_mean_matrix[i,3] = SUS
    overall_mean_matrix[i,4] = BE
    overall_mean_matrix[i,5] = GA
end

for i in 1:15
    (INF, REC,SUS,BE,GA) = std(foo[:,:,i], dims = 2)
    overall_std_matrix[i,1] = INF
    overall_std_matrix[i,2] = REC
    overall_std_matrix[i,3] = SUS
    overall_std_matrix[i,4] = BE
    overall_std_matrix[i,5] = GA
end


## Plots

step_vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]

#SIR plot
Gadfly.plot(layer(x = step_vec, y = overall_mean_matrix[:,1], Geom.line, Gadfly.Theme(default_color=color("red"))),
            layer(x = step_vec, y = overall_mean_matrix[:,2], Geom.line, Gadfly.Theme(default_color=color("blue"))),
            layer(x = step_vec, y = overall_mean_matrix[:,3], Geom.line, Gadfly.Theme(default_color=color("green"))),
            layer(x = step_vec, y = actuals , Geom.line, Gadfly.Theme(default_color=color("black"))),
            Guide.XLabel("Day"),
            Guide.YLabel("Population"),
            Guide.Title("Particle filter data compared to actual data"),
            Guide.manual_color_key("Legend", ["Susceptible", "Infected", "Recovered", "Actual"], ["red", "blue", "green", "black"]))

#evolution of beta and gamma plot
Gadfly.plot(layer(x = step_vec, y = overall_mean_matrix[:,4], Geom.line, Gadfly.Theme(default_color=color("red")), yintercept=[mean(overall_mean_matrix[:,4])],Geom.hline(style=:dot)),
            layer(x = step_vec, y = overall_mean_matrix[:,5], Geom.line, Gadfly.Theme(default_color=color("blue")), yintercept=[mean(overall_mean_matrix[:,5])],Geom.hline(style=:dot)),
            #layer(x = step_vec, y = mean(overall_mean_matrix[:,4]), Geom.line,Gadfly.Theme(default_color=color("black"))),
            #yintercept=[mean(overall_mean_matrix[:,4]), mean(overall_mean_matrix[:,5]), Geom.point, Geom.hline(style=:dot),
            Guide.XLabel("Day"),
            Guide.Title("Evolution of beta and gamma"),
            Guide.manual_color_key("Legend", ["Beta", "Gamma"], ["red", "blue"]))


##STD plots -> not too sure if right
# Gadfly.plot(layer(x = step_vec, y = overall_std_matrix[:,1], Geom.line, Gadfly.Theme(default_color=color("red"))),
#             layer(x = step_vec, y = overall_std_matrix[:,2], Geom.line, Gadfly.Theme(default_color=color("blue"))),
#             layer(x = step_vec, y = overall_std_matrix[:,3], Geom.line, Gadfly.Theme(default_color=color("green"))),
#             layer(x = step_vec, y = overall_std_matrix[:,4], Geom.line, Gadfly.Theme(default_color=color("black"))),
#             layer(x = step_vec, y = overall_std_matrix[:,5], Geom.line, Gadfly.Theme(default_color=color("pink"))),
#             Guide.XLabel("Day"),
#             Guide.Title("std of the different parameters"),
#             Guide.manual_color_key("Legend", ["Susceptible", "Infected", "Recovered", "Beta", "Gamma"], ["red", "blue", "green", "black", "pink"]))
