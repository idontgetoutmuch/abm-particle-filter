using Agents
using Random
# using DataFrames
# using LightGraphs
using Distributions
using DrWatson: @dict
using Plots
# using InteractiveDynamics
# using CairoMakie
using Statistics: mean
# using DifferentialEquations
# using SimpleDiffEq
# using StatsPlots


# Agent type
mutable struct Student <: AbstractAgent
    id::Int
    status::Symbol  # 1: S, 2: J, 3: I, 4: R
end

# Function to calculate gamma
function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end;

# Model function
function init_model(β :: Float64, c :: Float64, γ :: Float64, N :: Int64, I0 :: Int64)
    properties = @dict(β,c,γ)
    model = ABM(Student; properties=properties)
    for i in 1 : N
        if i <= I0
            s = :I
        else
            s = :S
        end
        p = Student(i,s)
        p = add_agent!(p,model)
    end
    return model
end;

# Agent step function
function agent_step!(agent, model)
    transmit!(agent, model)
    recover!(agent, model)
    develop!(agent, model)
end;

function transmit!(agent, model)
    agent.status != :S && return
    ncontacts = rand(Poisson(model.properties[:c]))
    for i in 1:ncontacts
        alter = random_agent(model)
        if alter.status == :I && (rand() ≤ model.properties[:β])
            agent.status = :J
            break
        end
    end
end;

function recover!(agent, model)
    agent.status != :I && return
    if rand() ≤ model.properties[:γ]
        agent.status = :R
    end
end;

function develop!(agent, model)
    agent.status != :J && return
    agent.status = :I
end;

# Model parameters
δt = 1.0
nsteps = 15
tf = nsteps * δt;
t = 0 : δt : tf;

β = 0.15
c = 7.5 * δt
γ = rate_to_proportion(0.50, δt);

N = 763
I0 = 1;

actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]

# ------------------
# Particle Filtering
# ------------------

Qbeta  = 1.0e-6
Qc     = 5.0e-7
Qgamma = 5.0e-7

Q = [Qbeta 0.00 0.00; 0.0 Qc 0.0; 0.0 0.0 Qgamma];
R = Matrix{Float64}(undef,1,1);
R[1,1] = 0.1;

P = 50;

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

function pf_init(inits, N, y, Q, R)

    log_w = zeros(N);
    y_pf = zeros(Int64,N);
    y_pf = map(x -> x[3], map(modelCounts, inits))
    wn = zeros(N);

    log_w = map(x -> logpdf(MvNormal([y], R), x), map(x -> [x], y_pf))

    wn = map(x -> exp(x), log_w .- maximum(log_w));
    wn = wn / sum(wn);

    return(y_pf, log_w, inits)

end

function pf(inits, log_w, N, y, Q, R)

    wn = zeros(N);
    jnits = [init_model(rand(LogNormal(log(β), Qbeta)), rand(LogNormal(log(c), Qc)), rand(LogNormal(log(γ), Qgamma)), N, 1) for n in 1:P]

    wn = map(x -> exp(x), log_w[:] .- maximum(log_w[:]));
    wn = wn / sum(wn);

    y_pf = zeros(Int64,N);

    a = resample_stratified(wn);

    for i in 1:N
        jnits[i] = deepcopy(inits[a[i]])
    end

    for i in 1:N
        Agents.step!(jnits[i], agent_step!, 1)
        currS, currJ, currI, currR = modelCounts(jnits[i])
        y_pf[i] = currI
    end

    epsilons = rand(MvNormal(zeros(3), Q), N)
    for i in 1:N
        jnits[i].properties[:β] = exp(log(jnits[i].properties[:β]) + epsilons[1,i])
        jnits[i].properties[:c] = exp(log(jnits[i].properties[:c]) + epsilons[2,i])
        jnits[i].properties[:γ] = exp(log(jnits[i].properties[:γ]) + epsilons[3,i])
    end

    log_w = map(x -> logpdf(MvNormal([y], R), x), map(x -> [x], y_pf))

    wn = map(x -> exp(x), log_w .- maximum(log_w));
    wn = wn / sum(wn);

    return(y_pf, log_w, jnits, a)

end

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



Random.seed!(1234);

templates = [init_model(rand(LogNormal(log(β), Qbeta)), rand(LogNormal(log(c), Qc)), rand(LogNormal(log(γ), Qgamma)), N, 1) for n in 1:P]

function runPf(inits, init_log_weights, predicted1, Q, R)
    l = length(actuals)
    predicted = zeros(l);
    predicted[1] = predicted1;

    for i in 2:l
        (end_states2, logWeights2, inits2, a2) = pf(deepcopy(inits), init_log_weights, P, map(x -> convert(Float64,x), actuals[i]), Q, R);
        predicted[i] = mean(end_states2);
        inits = inits2;
        init_log_weights = logWeights2;
    end
    return predicted;
end

function particleFilter(templates, P, actuals, Q, R)
    (initial_end_states, init_log_weights, inits) = pf_init(deepcopy(templates), P, map(x -> convert(Float64,x), actuals[1]), Q, R);
    predicted1 = mean(initial_end_states);
    predicted = runPf(inits, init_log_weights, predicted1, Q, R);
    return predicted;
end

@time faa = particleFilter(templates, P, actuals, Q, R);

Plots.plot(1:l, actuals, label="Actual", color = "red", lw = 3, title = string("Results from ", P, " runs"), xlab="Time",ylabel="Number")
Plots.plot!(1:l, predicted, label="Tracked by Particle Filter", color = "blue", lw = 3)

function pmh(inits, K, N, n_th, y, f_g, g, nx, prior_sample, prior_pdf, Q, R)

    T = length(y);
    theta = zeros(n_th, K+1);
    log_W = -Inf;
    # FIXME:
    x_pfs = zeros(nx, N, T, K);

    while log_W == -Inf # Find an initial sample without numerical problems
        theta[:, 1] = 9 .+ prior_sample(1);
        # FIXME:
        log_W = pf(inits, N, (x) -> f_g(x, theta[:, 1][1]), g, y, Q, R, nx)[3];
    end

    for k = 1:K
        theta_prop = map(exp, map(log, theta[:, k]) + 0.01 * rand(MvNormal(zeros(n_th), 1), 1)[1, :]);
        # log_W_prop = pf(inits, N, (x) -> f_g(x, theta_prop[1]), g, y, Q, R, nx)[3];
        (a, b, c) = pf(inits, N, (x) -> f_g(x, theta_prop[1]), g, y, Q, R, nx);
        log_W_prop = c;
        x_pfs[:, :, :, k] = a;
        mh_ratio = exp(log_W_prop - log_W) * prior_pdf(theta_prop) / prior_pdf(theta[:,k]);

        display([theta[:, k], theta_prop, log_W, log_W_prop, mh_ratio, prior_pdf(theta_prop)]);

        if isnan(mh_ratio)
            alpha = 0;
        else
            alpha = min(1,mh_ratio);
        end

        dm = rand();
        if dm < alpha
            theta[:, k+1] = theta_prop;
            log_W = log_W_prop;
            new = true;
        else
            theta[:, k+1] = theta[:, k];
            new = false;
        end

        # if new == true;
        #     display(["PMH Sampling ", k, ": Proposal accepted!"]);
        # else
        #     display(["PMH Sampling ", k, ": Proposal rejected"]);
        # end
    end
    return (x_pfs, theta);
end
