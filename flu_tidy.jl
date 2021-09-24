using Agents, Random, DataFrames, LightGraphs
using Distributions
using DrWatson: @dict
using Plots
using Random
using InteractiveDynamics
using CairoMakie
using Statistics: mean
using DifferentialEquations
using SimpleDiffEq
using StatsPlots





## agent type
mutable struct Student <: AbstractAgent
    id::Int
    status::Symbol  # 1: S, 2: J, 3: I, 4: R
end

## function to calculate gamma
function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end;



##model function
function init_model(β :: Float64, c :: Float64, γ :: Float64, N :: Int64, I0 :: Int64, G :: Vector{Int64})
    properties = @dict(β,c,γ,G)
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
    develop!(agent, model)
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
            # An pre-infection occurs
            agent.status = :J
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

function develop!(agent, model)
    #if agent is not pre-infected then return
    agent.status != :J && return
    #infected always becomes infected
    agent.status = :I
end;

##functions to determine S,I,R
susceptible(x) = count(i == :S for i in x)
pre_infected(x) = count(i == :J for i in x)
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x);

##model parameters
# δt = 0.1
δt = 1.0
# nsteps = 150
nsteps = 15
tf = nsteps * δt #t final
t = 0 : δt : tf; #time vector

β = 0.15
# β = 0.05
# β = 0.25
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
abm_model = init_model(β, c, γ, N, I0, G)

#collect data
to_collect = [(:status, f) for f in (susceptible, pre_infected, infected, recovered)]
abm_data, _ = run!(abm_model, agent_step!, nsteps; adata = to_collect);

abm_data[!,:t] = t;

Plots.plot(t,abm_data[:,2],label="S",xlab="Time",ylabel="Number")
Plots.plot!(t,abm_data[:,3],label="I")
Plots.plot!(t,abm_data[:,4],label="R")
Plots.plot!(1:15, actuals, label="A")

##run 50 times
model_1 = init_model(β, c, γ, N, I0, G);
agent_50_df, _ = run!(model_1, agent_step!, nsteps; adata = to_collect);
agent_50_df[!,:t] = t;

a_50 = agent_50_df[:,4];

# 151 element vector
# 16 element vector

for i in 2:50
    model_2 = init_model(β, c, γ, N, I0, G)
    agent_df_2, _ = run!(model_2,agent_step!,nsteps; adata = to_collect);

    a_1 = agent_df_2[:,4];
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

# ------------------
# Particle Filtering
# ------------------

Qbeta  = 1.0e-6
Qc     = 5.0e-7
Qgamma = 5.0e-7
# Parameter update covariace aka parameter diffusivity
Q = [Qbeta 0.00 0.00; 0.0 Qc 0.0; 0.0 0.0 Qgamma];


# Observation covariance
R = Matrix{Float64}(undef,1,1); #don't change
R[1,1] = 0.1;
# Number of particles
P = 50;
# P = 500;

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

function pf(inits, N, y, Q, R)

    T = length(y)
    log_w = zeros(T,N);
    y_pf = zeros(Int64,T,N);
    y_pf[1,:] = map(x -> x[2], map(modelCounts, inits))
    wn = zeros(N);

    for t = 1:T
        if t >= 2
            a = resample_stratified(wn);
            print("a = ", a, "\n")
            for i in 1:N
                print("i = ", i, " a[i] = ", a[i])
                inits[i].properties[:G][t] = a[i]
                print(" inits[i].properties[:G][t] = ", inits[i].properties[:G][t], "\n")
            end
            inits = inits[a]
            for i in 1:N
                print(" inits[i].properties[:G][t] = ", inits[i].properties[:G][t], "\n")
            end
            for i in 1:N
                # Agents.step!(inits[i], agent_step!, 10)
                Agents.step!(inits[i], agent_step!, 1)
                currS, currI, currR = modelCounts(inits[i])
                # print("props = ", inits[i].properties, " currS = ", currS, " currI = ", currI, " currR = ", currR, "\n")
                # print("currS = ", currS, " currI = ", currI, " currR = ", currR, " Total = ", currR + currI + currS, "\n")
                y_pf[t,i] = currI
            end

            epsilons = rand(MvNormal(zeros(3), Q), N)
            for i in 1:N
                inits[i].properties[:β] = exp(log(inits[i].properties[:β]) + epsilons[1,i])
                inits[i].properties[:c] = exp(log(inits[i].properties[:c]) + epsilons[2,i])
                inits[i].properties[:γ] = exp(log(inits[i].properties[:γ]) + epsilons[3,i])
            end
        end

        print("y[", t, "] = ", y[t], "\n")
        for i in 1:N
            print("y_pf[",t,",",i,"] = ", y_pf[t, i], "\n")
        end

        log_w[t, :] = map(x -> logpdf(MvNormal([y[t]], R), x), map(x -> [x], y_pf[t, :]))

        # To avoid underflow subtract the maximum before moving from
        # log space
        wn = map(x -> exp(x), log_w[t, :] .- maximum(log_w[t, :]));
        wn = wn / sum(wn);
        print("wn = ", wn, "\n")
    end

    log_W = sum(map(log, map(x -> x / N, sum(map(exp, log_w[1:T, :]), dims=2))));

    return(y_pf, log_w, log_W)

end

#creates P models with some variation in the params
G = zeros(Int64, 15)

inits = [init_model(rand(LogNormal(log(β), Qbeta)), rand(LogNormal(log(c), Qc)), rand(LogNormal(log(γ), Qgamma)), N, 1, copy(G)) for n in 1:P]

# (end_states, bar, baz) = pf(inits, P, map(x -> convert(Float64,x), actuals), Q, R);
(end_states, bar, baz) = pf(inits, P, map(x -> convert(Float64,x), actuals[1:4]), Q, R);

Function modelCounts(abm_model)
    nS = 0.0
    nI = 0.0
    nR = 0.0
    num_students = 763
    for i in 1:num_students
        status = get!(abm_model.agents, i, undef).status;
        if status == :S
            nS = nS + 1
        elseif status == :I
            nI = nI + 1
        else
            nR = nR + 1
        end
    end
    return nS, nI, nR
end
