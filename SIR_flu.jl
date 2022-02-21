#dependencies for this model
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
using ColorBrewer
using Compose
using Gtk


#Create Agent type
mutable struct Student <: AbstractAgent
    id::Int #id (required)
    status::Symbol  # 1: S, 2: J, 3: I, 4: R
end


#Function that initalises ABM and populates ABM with agents (including the one infected agent)
function init_model(model_params:: Vector{Float64}, N :: Int64, I0 :: Int64)
    #individually assign the values from input vector to their respective parameters
    β = model_params[1]
    c = model_params[2]
    γ = model_params[3]
    #create model properties
    properties = @dict(β,c,γ)
    #create ABM with the agent type and the properties
    model = ABM(Student; properties=properties)
    #For all 763 students
    for i in 1 : N
        if i <= I0 #for all initally infected agents
            s = :I #set status to infected
        else
            s = :S #else set status to suspectible
        end
        #create student with the status above and add agent to simulation
        p = Student(i,s)
        p = add_agent!(p,model)
    end
    return model
end;


#Function describes the behaviour of agents within a single time step
function agent_step!(agent, model)
    transmit!(agent, model)
    recover!(agent, model)
    develop!(agent, model)
end;


#Function describes the transmission of the flu within model
function transmit!(agent, model)
    #if agent is susceptiable, no infection occurs
    agent.status != :S && return
    #else, we determine a number of contacts
    ncontacts = rand(Poisson(model.properties[:c]))
    #loop through number of contacts
    for i in 1:ncontacts
        #select a random agent within simulation
        alter = random_agent(model)
        #if random agent is infected and meets condition
        if alter.status == :I && (rand() ≤ model.properties[:β])
            #then in transition state (to avoid infection and recovery in one time step)
            agent.status = :J
            break
        end
    end
end;


#Function describes the recovery from flu within model
function recover!(agent, model)
    #if agent is not infected, then no recovery occurs
    agent.status != :I && return
    #else recovery based on gamma
    if rand() ≤ model.properties[:γ]
        agent.status = :R
    end
end;

#Function describes the develop step (from J to I) within model
function develop!(agent, model)
    #if agent not in transition step, then no development occurs
    agent.status != :J && return
    #else, infect the agent in state J
    agent.status = :I
end;


susceptible(x) = count(i == :S for i in x)
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x);


function rate_to_proportion(r::Float64,t::Float64)
    #rate based on 1-e^(-r*t)
    1-exp(-r*t)
end;


Random.seed!(1234); #seed the RNG
δt = 1.0; #change in time
β  = 0.15; #rate of infection
c  = 7.5 * δt; #contact rate
γ  = rate_to_proportion(0.50, δt); #rate of recovery
N  = 763; #total number of students
I0 =   1; #one person infected initially


nsteps = 15 #number of time steps
tf = nsteps * δt;
t = 0 : δt : tf; #time vector


#define vector of parameters for init_model function
ABM_input = [β; c; γ]

#create model
abm_model = init_model(ABM_input, N, I0)

#collect data
to_collect = [(:status, f) for f in (susceptible, infected, recovered)]
abm_data, _ = run!(abm_model, agent_step!, nsteps; adata = to_collect);

#time step information
abm_data[!,:t] = t;


actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5] #data from paper (see intro)
Plots.plot(t,abm_data[:,2],label="S",xlab="Time",ylabel="Number", title = "One simulation of SIR model")
Plots.plot!(t,abm_data[:,3],label="I")
Plots.plot!(t,abm_data[:,4],label="R")
Plots.plot!(1:15, actuals, label="Actual")

##for code that runs model multiple times, see particle filter SIR ABM notebook

## particle filter functions

function modelCounts(abm_model)
    nS = 0.0
    nJ = 0.0
    nI = 0.0
    nR = 0.0
    num_students = 763
    for i in 1:num_students
        status = get!(abm_model.agents, i, undef).status;
        if status == :S
            nS = nS + 1
        elseif status == :J
            nJ = nJ + 1
        elseif status == :I
            nI = nI + 1
        else
            nR = nR + 1
        end
    end
    return nS, nJ, nI, nR
end


# Parameters to be estimated
μ = [β, c, γ];

Qbeta  = 0.002 #covariance for β
Qc     = 0.005 #covariance for c
Qgamma = 0.002 #covariance for γ

# Parameter update covariance aka parameter diffusivity
var = [[Qbeta, 0.0, 0.0] [0.0,  Qc , 0.0] [0.0,  0.0, Qgamma]];

# Observation covariance
R = Matrix{Float64}(undef,1,1);
R[1,1] = 0.1;

# Number of particles
P = 50;

#initalise the various models
input_vec = [β; c; γ]
templates = [init_model(input_vec, N, 1) for n in 1:P]


function resample_stratified(weights)

    N = length(weights)
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


function measure(x)
    return modelCounts(x)[3]
end


function pf_init(inits, g, N, y, R)

    y_pf = zeros(Int64,N);
    log_w = zeros(N);

    y_pf = map(g, inits)
    log_w = map(x -> logpdf(MvNormal([y], R), x), map(x -> [x], y_pf))

    return(y_pf, log_w)

end


function pf(inits, g, log_w, N, y, R)

    wn = zeros(N);
    in_vec_1 = [β; c; γ]
    jnits = [init_model(in_vec_1, N, 1) for n in 1:P]
    y_pf = zeros(Int64,N);

    wn = exp.(log_w .- maximum(log_w));
    swn = sum(wn);
    wn .= wn ./ swn;

    a = resample_stratified(wn);

    for i in 1:N
        jnits[i] = deepcopy(inits[a[i]])
    end

    for i in 1:N
        Agents.step!(jnits[i], agent_step!, 1)
        y_pf[i] = g(jnits[i])
    end

    log_w = map(x -> logpdf(MvNormal([y], R), x), map(x -> [x], y_pf))

    max_weight = maximum(log_w);
    wn = exp.(log_w .- max_weight);
    swn = sum(wn);
    wn .= wn ./ swn;
    predictive_likelihood = max_weight + log(swn) - log(N);

    return(y_pf, log_w, predictive_likelihood, jnits)

end


function runPf(inits, g, init_log_weights, actuals, predicted1, R)
    l         = length(actuals)
    predicted = zeros(l);
    log_w     = zeros(l, P);

    predicted[1] = predicted1;
    log_likelihood = 0;

    for i in 2:l
        (obs, new_log_weights, predictive_likelihood, news) = pf(deepcopy(inits), g, init_log_weights, P, map(x -> convert(Float64,x), actuals[i]), R);
        predicted[i] = mean(obs);
        log_likelihood = log_likelihood + predictive_likelihood;

        inits            = news;
        init_log_weights = new_log_weights;
    end
    return predicted, log_likelihood;
end


function particleFilter(templates, g, P, actuals, R)
    l = length(actuals);

    inits = deepcopy(templates);
    (initial_end_states, init_log_weights) = pf_init(inits, g, P, map(x -> convert(Float64,x), actuals[1]), R);

    predicted, log_likelihood = runPf(inits, g, init_log_weights, actuals, mean(initial_end_states), R);

    return predicted, log_likelihood;
end


@time predicted = particleFilter(templates, measure, P, actuals, R);

l = length(actuals);

Plots.plot(1:l, actuals, label="Actual", color = "red", lw = 3, title = string("Results from ", P, " particles\n"), xlab="Time",ylabel="Number", legend = :left)
Plots.plot!(1:l, predicted, label="Tracked by Prior Particle Filter", color = "blue", lw = 3)


function prior_sample(μ, var)
    rand(MvLogNormal(log.(μ), var))
end

function log_prior_pdf(x, μ, var)
    logpdf(MvLogNormal(log.(μ), var), x)
end


function pmh(g, P, N, K, μ, var, actuals, R,num_params, theta)
    prop_acc            = zeros(K);
    log_likelihood_curr = -Inf;

    log_prior_curr = log_prior_pdf(theta[:, 1], μ, var);

    for k = 2:K
        theta_prop  = prior_sample(theta[:, k - 1], var);
        theta[:, k] = theta_prop;
        # β = theta[1, k];
        # c = theta[2, k];
        # γ = theta[3, k];

        inits = [init_model(theta[:,k], N, 1) for n in 1:P];
        predicted, log_likelihood_prop = particleFilter(inits, g, P, actuals, R);
        log_likelihood_diff = log_likelihood_prop - log_likelihood_curr;

        log_prior_curr = log_prior_pdf(theta[:, k - 1], μ, var);
        log_prior_prop = log_prior_pdf(theta[:, k],     μ, var);
        log_prior_diff = log_prior_prop - log_prior_curr;

        acceptance_prob = exp(log_prior_diff + log_likelihood_diff);

        r = rand();
        if (r < acceptance_prob)
            log_likelihood_curr = log_likelihood_prop;
            prop_acc[k]         = 1;
        else
            theta[:, k] = theta[:, k - 1];
            prop_acc[k] = 0;
        end

        print("#####################################################################\n");
        print(" Iteration: ", k, " completed.\n");
        print(" Current state of the Markov chain: ", theta[:, k], "\n");
        print(" Proposed state of the Markov chain: ", theta_prop, "\n");
        print(" Current posterior mean: ", mean(theta[:, 1:k], dims = 2), "\n");
        print(" Current acceptance: ", mean(prop_acc[1:k]), "\n");
        print("#####################################################################\n");


    end
end


Random.seed!(1234);

K = 100 #Number of posterior particle filter runs
num_params = 3; #number of parameters to be estimated
theta = zeros(num_params, K); #initalised theta or parameter vector
theta[:, 1] = prior_sample(μ, var); #detered parameters

#run pmh function
pmh(measure, P, N, K, μ, var, actuals, R, num_params, theta)


# This mean of the posterior was used
(β, c, γ) = [0.14455132434154833; 9.617284697797249; 0.4849730067859164]


mean_input = [β; c; γ]
templates = [init_model(mean_input, N, 1) for n in 1:P]

@time predicted_posterior = particleFilter(templates, measure, P, actuals, R);

Plots.plot(1:l, actuals, label="Actual", color = "red", lw = 3, title = string("Results from ", P, " particles and ", K, " Monte Carlo steps\n"), xlab="Time",ylabel="Number", legend = :left)
Plots.plot!(1:l, predicted, label="Tracked by Prior Particle Filter", color = "blue", lw = 3)
Plots.plot!(1:l, predicted_posterior, label="Tracked by Posterior Particle Filter", color = "green", lw = 3)
