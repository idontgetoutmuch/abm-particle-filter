using Agents
using Random
using Distributions
using DrWatson: @dict
using Plots
using Statistics: mean


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
function init_model(model_params:: Vector{Float64}, N :: Int64, I0 :: Int64)
    β = model_params[1]
    c = model_params[2]
    γ = model_params[3]
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
#infected(x) = count(i == :I for i in x)


#infected(a) = a.status == :I
infected(m) = count(a.status == :I for a in allagents(m))
function measure(x)
    num_infec = infected(x);
    return num_infec
    #return modelCounts(x)[3]
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

    # Can this be made parallel?
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

# Finally we can run the particle filter
Random.seed!(1234);

δt = 1.0;
β  = 0.15;
c  = 7.5 * δt;
γ  = rate_to_proportion(0.50, δt);

N  = 763;
I0 =   1;

actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5];

R = Matrix{Float64}(undef,1,1);
R[1,1] = 0.1;

P = 50;
input_vec = [β; c; γ]
templates = [init_model(input_vec, N, 1) for n in 1:P]

@time predicted = particleFilter(templates, measure, P, actuals, R);

l = length(actuals);

Plots.plot(1:l, actuals, label="Actual", color = "red", lw = 3, title = string("Results from ", P, " particles\n"), xlab="Time",ylabel="Number", legend = :left)
Plots.plot!(1:l, predicted, label="Tracked by Prior Particle Filter", color = "blue", lw = 3)

# Parameters to be estimated
μ = [β, c, γ];
var = [[0.002, 0.0, 0.0] [0.0,  0.005, 0.0] [0.0,  0.0, 0.002]];

function prior_sample(μ, var)
    rand(MvLogNormal(log.(μ), var))
end

function log_prior_pdf(x, μ, var)
    logpdf(MvLogNormal(log.(μ), var), x)
end


# Should prior_sample and log_prior_pdf be paramaters and passed into
# pmh rather than being global?
function pmh(g, P, N, K, μ, var, actuals, R,num_params, theta)
    # This need generalising - in this case we have 3 parameters but
    # we should handle any number
    prop_acc            = zeros(K);
    log_likelihood_curr = -Inf;

    # FIXME: Is this really needed now we are using the technique
    # here:
    # https://github.com/compops/pmh-tutorial/blob/master/matlab/particleFilter.m#L47
    # while log_likelihood_curr == -Inf
    #     theta[:, 1] = prior_sample(μ, var);
    #     β = theta[1, 1];
    #     c = theta[2, 1];
    #     γ = theta[3, 1];
    #
    #     inits = [init_model(β, c, γ, N, 1) for n in 1:P];
    #     predicted, log_likelihood_curr = particleFilter(inits, g, P, actuals, R);
    # end

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

K = 100
num_params = 3;
theta = zeros(num_params, K);
theta[:, 1] = prior_sample(μ, var);


pmh(measure, P, N, K, μ, var, actuals, R, num_params, theta)

Random.seed!(1234);

# This mean of the posterior was used
(β, c, γ) = [0.14455132434154833; 9.617284697797249; 0.4849730067859164]

mean_input = [β; c; γ]
templates = [init_model(mean_input, N, 1) for n in 1:P]

@time predicted_posterior = particleFilter(templates, measure, P, actuals, R);

Plots.plot(1:l, actuals, label="Actual", color = "red", lw = 3, title = string("Results from ", P, " particles and ", K, " Monte Carlo steps\n"), xlab="Time",ylabel="Number", legend = :left)
Plots.plot!(1:l, predicted, label="Tracked by Prior Particle Filter", color = "blue", lw = 3)
Plots.plot!(1:l, predicted_posterior, label="Tracked by Posterior Particle Filter", color = "green", lw = 3)

# ------------------
# Particle Filtering
# ------------------

Qbeta  = 1.0e-6
Qc     = 5.0e-7
Qgamma = 5.0e-7

Q = [Qbeta 0.00 0.00; 0.0 Qc 0.0; 0.0 0.0 Qgamma];
R = Matrix{Float64}(undef,1,1);
R[1,1] = 0.1;

nsteps = 15
tf = nsteps * δt;
t = 0 : δt : tf;
