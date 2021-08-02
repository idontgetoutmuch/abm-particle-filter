using Agents, Random, DataFrames, LightGraphs
using Distributions: Poisson, DiscreteNonParametric
using DrWatson: @dict
using Plots
using Random
using InteractiveDynamics
using CairoMakie
using Statistics: mean
using DifferentialEquations
using SimpleDiffEq
using StatsPlots

using Distributions


function sir_ode!(du,u,p,t)
    (S, I, R) = u
    (β, γ)    = p
    N = S + I + R
    @inbounds begin
        du[1] = -β * I / N * S
        du[2] =  β * I / N * S - γ * I
        du[3] =                  γ * I
    end
    nothing
end;


δt = 0.1
tmax = 15.0
tspan = (0.0, tmax)
t = 1.0 : δt : tmax;

u0 = [762.0, 1.0, 0.0]; # S, I, R
p = [2.0, 0.5]; # β, γ
actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]


prob_ode = ODEProblem(sir_ode!, u0, tspan, p);
sol_ode = solve(prob_ode);

df_ode = DataFrame(Tables.table(sol_ode(t)'))
df_ode[!,:t] = t;

p1 = [1.7, 0.5]; # β, γ

prob_ode1 = ODEProblem(sir_ode!, u0, tspan, p1);
sol_ode1 = solve(prob_ode1);

df_ode1 = DataFrame(Tables.table(sol_ode1(t)'))
df_ode1[!,:t] = t;

Plots.plot(df_ode1[:,:t],df_ode1[:,1],label="S",xlab="Time",ylabel="Number", title = "ODE solution")
Plots.plot!(df_ode1[:,:t],df_ode1[:,2],label="I")
Plots.plot!(df_ode1[:,:t],df_ode1[:,3],label="R")
Plots.plot!(1:15, actuals, label="A")

## agent type
mutable struct Student <: AbstractAgent
    id::Int
    status::Symbol  # 1: S, 2: I, 3:R
end

## function to calculate gamma
function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end;



##model function
function init_model(β :: Float64, c :: Float64, γ :: Float64, N :: Int64, I0 :: Int64)
    properties = @dict(β,c,γ)
    model = ABM(Student; properties=properties)
    for i in 1 : N #for all students
        if i <= I0 # infect initial number of students
            s = :I
        else
            s = :S # rest of students are susceptible
        end
        p = Student(i,s) #create student agent type
        p = add_agent!(p,model) #add this student to ABM
    end
    return model
end;

##Agent step function

function agent_step!(agent, model)
    transmit!(agent, model)
    recover!(agent, model)
end;

function transmit!(agent, model)
    # If I'm not susceptible, I return
    agent.status != :S && return
    #based on c value, decide contacts
    ncontacts = rand(Poisson(model.properties[:c]))
    for i in 1:ncontacts
        # Choose random individual
        alter = random_agent(model)
        # if random agent is infected, and rand is <=
        # beta (infection rate)
        if alter.status == :I && (rand() ≤ model.properties[:β])
            # An infection occurs
            agent.status = :I
            break
        end
    end
end;


function recover!(agent, model)
    #if agent is not infected, then returns
    agent.status != :I && return
    #if random probability of recovery <= gamma
    if rand() ≤ model.properties[:γ]
        #then recover agent
            agent.status = :R
    end
end;

##functions to determine S,I,R
susceptible(x) = count(i == :S for i in x)
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x);

##model parameters
δt = 0.1
nsteps = 150
tf = nsteps * δt #t final
t = 0 : δt : tf; #time vector

# β = 0.05
β = 0.25
# c = 10.0 * δt
c = 7.5 * δt #contact rate
# γ = rate_to_proportion(0.25, δt);
γ = rate_to_proportion(0.50, δt); #determine gamma

# N = 1000
N = 763 #total number of students
# I0 = 10;
I0 = 1; #one person infected

actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]

Random.seed!(1234);

#create model
abm_model = init_model(β, c, γ, N, I0)

#collect data
to_collect = [(:status, f) for f in (susceptible, infected, recovered)]
abm_data, _ = run!(abm_model, agent_step!, nsteps; adata = to_collect);

abm_data[!,:t] = t;

Plots.plot(t,abm_data[:,2],label="S",xlab="Time",ylabel="Number")
Plots.plot!(t,abm_data[:,3],label="I")
Plots.plot!(t,abm_data[:,4],label="R")
Plots.plot!(1:15, actuals, label="A")

##
# try to run for a single time step only




##run 50 times
model_1 = init_model(β, c, γ, N, I0);
agent_50_df, _ = run!(model_1, agent_step!, nsteps; adata = to_collect);
agent_50_df[!,:t] = t;

a_50 = agent_50_df[:,3];

# 151 element vector

for i in 2:50
    model_2 = init_model(β, c, γ, N, I0)
    agent_df_2, _ = run!(model_2,agent_step!,nsteps; adata = to_collect);

    a_1 = agent_df_2[:,3];
    global a_50 = [a_50 a_1]
end

Plots.plot(1:15, actuals, label="Actual", color = "red", lw = 3, legend = :outertopleft)
for i in 1:49
    Plots.plot!(t, a_50[:,i])#, seriestype = :scatter)
end
Plots.plot!(t,a_50[:,50])#, seriestype = :scatter)


## plotting the mean, std, median
# calculate mean, std, median of all runs
mean_of_50 = mean(a_50, dims = 2)
median_of_50 = median(a_50, dims = 2)
std_of_50 = std(a_50, dims = 2)

Plots.plot(1:15, actuals, label="Actual", color = "red", lw = 3, title = "Results from 50 runs", xlab="Time",ylabel="Number")
Plots.plot!(t, mean_of_50, label = "Mean", lw = 2)
Plots.plot!(t, median_of_50, label = "Median", lw = 2)
Plots.plot!(t, 2*std_of_50 + mean_of_50, label = "2+ std", lw = 2)
Plots.plot!(t, abs.(mean_of_50 - 2*std_of_50) , label = "2- std", lw = 2)



## Particle filter

##using save method
model_3 = init_model(β, c, γ, N, I0)

#collect data
to_collect = [(:status, f) for f in (susceptible, infected, recovered)]
abm_data_3, _ = run!(model_3, agent_step!, 0; adata = to_collect);
AgentsIO.save_checkpoint("one_step.jld2", model_3)
model = AgentsIO.load_checkpoint("one_step.jld2"; scheduler = Schedulers.randomly)
abm_data_2, _ = run!(model, agent_step!, 0; adata = to_collect);

total_infected(m) = count(a.status == :I for a in allagents(m))
total_recovered(m) = count(a.status == :R for a in allagents(m))
total_sus(m) = count(a.status == :S for a in allagents(m))


has_state(agent, status) = agent.status == status

## using random agent
model_1 = init_model(β, c, γ, N, I0)

total = zeros(5)

for i in 1:5
    agent = random_agent(model_1)
    agent_step!(agent,model_1)
    total[i] = total_infected(model_1)
end

Plots.plot(1:15, total)



##using the original but with one_step

condition(status) = agent -> has_state(agent, status)
random_agent(model, condition(:S))


#new functions that are not agent dependent
function one_step!(model)
    transmit_one_step!(model)
    recover_one_step!(model)
end;

function transmit_one_step!(model)
    # If I'm not susceptible, I return
    agent = random_agent(model)
    agent.status != :S && return
    #based on c value, decide contacts
    ncontacts = rand(Poisson(model.properties[:c]))
    for i in 1:ncontacts
        # Choose random individual
        alter = random_agent(model)
        # if random agent is infected, and rand is <=
        # beta (infection rate)
        if alter.status == :I && (rand() ≤ model.properties[:β])
            # An infection occurs --> instead of infecting the agent,
            # a random agent is infected instead
            another_agent = random_agent(model, condition(:R))
            if isnothing(another_agent)
                break
            else
                another_agent.status = :I
                break
            end
        end
    end
end;


function recover_one_step!(model)
    agent = random_agent(model)
    #if agent is not infected, then returns
    agent.status != :I && return
    #if random probability of recovery <= gamma
    if rand() ≤ model.properties[:γ]
        #then recover agent
            agent.status = :R
    end
end;



##using step!
model_5 = init_model(β, c, γ, N, I0)
Agents.step!(model_5,agent_step!)



total = zeros(15)

for i in 1:15
    Agents.step!(model_5, agent_step!)
    total[i] = total_infected(model_5)
end


##

function n(model,step)
end

model_6 = init_model(β, c, γ, N, I0)
Agents.step!(model_6, agent_step!, dummystep,n(model_6,2))


##

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
    c = vec[5]
    gamma = vec[6]







    ##returns int value of number infected
    # new_infected = rand(rng, Poisson(oldS * oldI * beta / num_students))
    # valid_infected = min(new_infected, oldS)

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
#3x3 matrix
#match beta, c 0.005, gamma, 3x1.75 7.65
Q = [0.01 0.00 0.00; 0.0 0.005 0.0; 0.0 0.0 0.005]; #--> spread is ok, peak is not
#Q = [0.01 0.0; 0.0 0.01]; #--> spread does not change, peak is flattened
#Q = [0.001 0.0; 0.0 0.001]; #--> spread effected, peak is flattened
#Q = [0.005 0.0; 0.0 0.05]; #--> does not reflect spread

# Observation covariance
R = Matrix{Float64}(undef,1,1); #don't change
R[1,1] = 0.1;
# Number of particles
N = 10000; #1000;
# S, I, R beta and gamma
nx = 6;
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
# add in the c
inits = zeros(nx, N)
inits[1, :] .= 762;
inits[2, :] .= 1;
inits[3, :] .= 0;
inits[4, :] .= rand(LogNormal(log(2),0.01),N)
inits[5, :] .= rand(LogNormal(log(0.75),0.005),N);
inits[6, :] .= rand(LogNormal(log(0.5),0.005),N);



#actual values
y = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]


(end_states, bar, baz) = pf(inits, N, simulate_one_step, observe, y, Q, R, nx, ny);





#
#
#
#
#
#
#
#
#
#
#
#
#
# ## black box optimise
# # function myCost(x, ys)
# #     model = init_model(
# #         x[1],
# #         x[2],
# #         x[3],
# #         N,
# #         I0,
# #
# #     )
# #
# #     _, data = run!(model,agent_step!,20;
# #                     adata = [infected], when_model = collect(1:14),
# #                    replicates = 1,)
# #     s = 0
# #     for i = 1:14
# #         s = s + (data.total_infected[i] - ys[i])^2
# #     end
# #     return Float64(s)
# # end
# #
# # #define part cost function
# # partCost(ys) = x -> myCost(x, ys)
# #
# # Random.seed!(10)
# #
# # #x0 with values used in simulation
# # x0 = [
# #     2.0,
# #     0.75,
# #     0.5,
# #     N,
# #     I0,
# # ]
# #
# # #actual values
# # actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]
# #
# #
# #
# # #part cost of actuals vs simulation values
# # partCost(actuals)(x0)
# #
# #
# # # optimise result
# # result = bboptimize(partCost(actuals),
# #     SearchRange = [
# #         (1.0, 10.0),
# #         (0.5,1.0),
# #         (0.1, 1.0),
# #    ],
# #    NumDimensions = 3,
# #    MaxTime = 50,
# #                     )
# #
# #  best_fitness(result)
# #
# #  x = best_candidate(result)
# #
# #  Random.seed!(10)
# #
# # #model generated from optimised values
# #  # model_op = init_model(;
# #  #     β = x[1],
# #  #     c = x[2],
# #  #     γ = x[3],
# #  # )
# #
# # model_op = init_model(β = x[1], c = x[2], γ = x[3], N, I0);
# # op_data, _ = run!(model_op, agent_step!, nsteps; adata = to_collect);
