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
N = 1000; #1000;
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

## Plots of mean sus, infected, recovered, beta and gamma and std --> based on time step

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

#infceted, infected + 1 std and infected - 1 std




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


#STD plots -> not too sure if right
Gadfly.plot(layer(x = step_vec, y = actuals, Geom.line, Gadfly.Theme(default_color=color("red"))),
            layer(x = step_vec, y = 2*overall_std_matrix[:,2] + overall_mean_matrix[:,2]  , Geom.line, Gadfly.Theme(default_color=color("blue"))),
            layer(x = step_vec, y = overall_mean_matrix[:,2], Geom.line, Gadfly.Theme(default_color=color("green"))),
            layer(x = step_vec, y = overall_mean_matrix[:,2] - 2*overall_std_matrix[:,2] , Geom.line, Gadfly.Theme(default_color=color("black"))),
            #layer(x = step_vec, y = overall_std_matrix[:,5], Geom.line, Gadfly.Theme(default_color=color("pink"))),
            # Guide.XLabel("Day"),
            Guide.XLabel("Day"),
            Guide.Title("std of the different parameters"),
            Guide.manual_color_key("Legend", ["actuals", "Inf + 1 ", "Infected", "infect - 1"], ["red", "blue", "green", "black"]))
            # Guide.Title("std of the different parameters"),
            # Guide.manual_color_key("Legend", ["Infected"], ["red"]))
            # Guide.XLabel("Day"),
            # Guide.Title("std of the different parameters"),
            # Guide.manual_color_key("Legend", ["Susceptible", "Infected", "Recovered", "Beta", "Gamma"], ["red", "blue", "green", "black", "pink"]))

# Plots of all particles
lines = [layer(x= step_vec,y= foo[2,i,:], Geom.line,Gadfly.Theme(line_width = 0.6mm)) for i in range(1,stop=50)] #change this for more paths
actuals_line = [layer(x = step_vec, y = actuals, Geom.line, Gadfly.Theme(default_color=color("red"),line_width = 1mm))]
two_plus_std_line = [layer(x = step_vec, y = 2*overall_std_matrix[:,2] + overall_mean_matrix[:,2]  , Geom.line, Gadfly.Theme(default_color=color("black"),line_width = 0.7mm))]
two_minus_std_line = [layer(x = step_vec, y = overall_mean_matrix[:,2] - 2*overall_std_matrix[:,2] , Geom.line, Gadfly.Theme(default_color=color("green"), line_width = 0.7mm))]
mean_line = [layer(x = step_vec, y = overall_mean_matrix[:,2], Geom.line, Gadfly.Theme(default_color=color("blue"), line_width = 0.8mm))]

#append!(actuals_line,lines)
append!(actuals_line,mean_line)
append!(actuals_line, two_plus_std_line)
append!(actuals_line,two_minus_std_line)
append!(actuals_line,lines)

#all plots
Gadfly.plot(actuals_line...,Coord.Cartesian(ymin=-0.1,ymax=700), Guide.XLabel("Day"),Guide.YLabel("Population"), Guide.Title("Plot of infected individuals overtime"), Guide.manual_color_key("Legend", ["particle paths","Actual data", "Mean data", "2+ std", "2- std"], ["deepskyblue", "red", "blue", "black", "green"]))

# only paths of particles
Gadfly.plot(lines..., Guide.XLabel("Day"),Guide.YLabel("Population"),Guide.Title("Plot of infected individuals overtime"), Guide.manual_color_key("Legend", ["particle paths"], ["deepskyblue"]))


## Histograms
#variation for each param, at all time steps
col = Colors.distinguishable_colors(15)

sus_variation = [layer(x= foo[1,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]
inf_variation = [layer(x= foo[2,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]
rec_variation = [layer(x= foo[3,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]
beta_variation = [layer(x= foo[4,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]
gamma_variation = [layer(x= foo[5,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]

p1 = Gadfly.plot(sus_variation..., Guide.Title("Susceptible"))
p2 = Gadfly.plot(inf_variation..., Guide.Title("Infected"))
p3 = Gadfly.plot(rec_variation...,Guide.Title("Recovered"))
p4 = Gadfly.plot(beta_variation..., Guide.Title("Beta"))
p5 = Gadfly.plot(gamma_variation..., Guide.Title("Gamma"))

gridstack(Union{Plot,Compose.Context}[p1 p2 p3; p4 p5 Compose.context()])



vstack(hstack(p1,p2), p3)
vstack(p1,p2)
vstack(p4,p5)
#hstack(p1, p2, p3)
#gridstack([p4 ; p5])



#p1 = Gadfly.plot(x = foo[4,:,1], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=color("black")))





##
# -------------------------------------
# Particle Marginal Metropolis Hastings
# -------------------------------------

function prior_pdf(theta)
    pdf(MvLogNormal(zeros(length(theta)), Matrix{Float64}(I, length(theta), length(theta))), theta)
end

function prior_sample(n)
    rand(MvLogNormal(zeros(n), Matrix{Float64}(I, n, n)))
end

function pmh(inits, K, N, n_th, y, f_g, g, nx, prior_sample, prior_pdf, Q, R)

    T = length(y);
    theta = zeros(n_th, K+1);
    log_W = -Inf;
    # FIXME:
    x_pfs = zeros(nx, N, T, K);

    while log_W == -Inf # Find an initial sample without numerical problems
        theta[:, 1] = map(exp, rand(MvNormal([log(2.0); log(0.5)], [0.1 0.0; 0.0 0.01] * Matrix{Float64}(I, 2, 2))));
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
