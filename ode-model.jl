using DifferentialEquations
using DataFrames
using Plots
using Distributions
using Random

function sir_ode!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    @inbounds begin
        du[1] = -β*c*I/N*S
        du[2] = β*c*I/N*S - γ*I
        du[3] = γ*I
    end
    nothing
end;

δt = 0.1
tmax = 14.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;

u0 = [763.0,1.0,0.0]; # S,I.R

# N.B. β and c are non-identifiable as we only ever use β * c
p = [0.2,10.0,0.5]; # β,c,γ

prob_ode = ODEProblem(sir_ode!,u0,tspan,p);

sol_ode = solve(prob_ode);

ys = [sol_ode(t) for t in 0.0:14]

df_ode = DataFrame(mapreduce(permutedims, vcat, ys), :auto)
df_ode[!,:t] = 0.0:14;

actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5];

df_ode[!,:actuals] = actuals;

plot(df_ode[!, :t],  df_ode[!, :x1], xaxis = "Time", yaxis = "Pupils", lw = 3, label = "Susceptible")
plot!(df_ode[!, :t], df_ode[!, :x2],                                   lw = 3, label = "Infected")
plot!(df_ode[!, :t], df_ode[!, :x3],                                   lw = 3, label = "Recovered")
plot!(df_ode[!, :t], df_ode[!, :actuals],                              lw = 3, label = "Actuals")

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

function modelCounts(ode_model)
    nS = ode_model[4]
    nI = ode_model[5]
    nR = ode_model[6]
    return nS, NaN, nI, nR
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

var = [[0.002, 0.0, 0.0] [0.0,  0.005, 0.0] [0.0,  0.0, 0.002]];

function pf(inits, g, log_w, P, y, R)

    wn = zeros(P);
    jnits = [init_model(NaN, NaN, NaN, NaN, NaN) for n in 1:P]
    y_pf = zeros(Float64,P);

    wn = exp.(log_w .- maximum(log_w));
    swn = sum(wn);
    wn .= wn ./ swn;

    a = resample_stratified(wn);

    for i in 1:P
        jnits[i] = deepcopy(inits[a[i]])
    end

    # Can this be made parallel?
    # for i in 1:N
    #     Agents.step!(jnits[i], agent_step!, 1)
    #     y_pf[i] = g(jnits[i])
    # end

    for i in 1:P
        prob_ode = ODEProblem(sir_ode!, jnits[i,:][1][4:6], (0.0, 1.0), jnits[i,:][1][1:3]);
        sol_ode = solve(prob_ode);
        foo = rand(MvLogNormal(log.(last(sol_ode.u)), var));
        jnits[i,:][1][4:6] = foo
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

N  = 763;
I0 =   1;

R = Matrix{Float64}(undef,1,1);
R[1,1] = 0.1;

P = 50;

# In ODE terms a model is the state and the parameters
function init_model(β, c, γ, N, I0)
    return [β; c; γ; N; I0; 0.0]
end

Random.seed!(1234);

templates = [init_model(p[1], p[2], p[3], N, 1) for n in 1:P]

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

var2 = [[0.002, 0.0] [0.0, 0.002]];

# Should prior_sample and log_prior_pdf be paramaters and passed into
# pmh rather than being global?
function pmh(g, P, N, K, μ, var, actuals, R)
    M                   = length(μ);
    (S, T)              = size(var);
    @assert M == S == T;
    theta               = zeros(M, K);
    prop_acc            = zeros(K);
    log_likelihood_curr = -Inf;

    # FIXME: Is this really needed now we are using the technique
    # here:
    # https://github.com/compops/pmh-tutorial/blob/master/matlab/particleFilter.m#L47
    while log_likelihood_curr == -Inf
        theta[:, 1] = prior_sample(μ, var);
        β = theta[1, 1];
        c = 10.0;
        γ = theta[2, 1];

        inits = [init_model(β, c, γ, N, 1) for n in 1:P];
        predicted, log_likelihood_curr = particleFilter(inits, g, P, actuals, R);
    end

    log_prior_curr = log_prior_pdf(theta[:, 1], μ, var);

    for k = 2:K
        theta_prop  = prior_sample(theta[:, k - 1], var);
        theta[:, k] = theta_prop;
        β = theta[1, k];
        c = theta[2, k];
        γ = theta[3, k];

        inits = [init_model(β, c, γ, N, 1) for n in 1:P];
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
