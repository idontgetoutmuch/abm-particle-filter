using Agents, Random, DataFrames, LightGraphs
using Distributions: Poisson, DiscreteNonParametric, MvNormal
using DrWatson: @dict
using Plots
using Random
using InteractiveDynamics
using CairoMakie
using BlackBoxOptim, Random
using Statistics: mean
using Gadfly




#how many students
#num_students = 763

#Agent type
mutable struct Student <: AbstractAgent
    id::Int
    status::Symbol  # 1: S, 2: I, 3:R
end

function model_initialize(; num_students = 763, beta = 2, gamma = 0.5, seed = 125)
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
    new_infected = rand(rng,Poisson(total_suspec * total_inf * beta_param / 763))

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


function plot_timeseries()
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Number infected")
    infectedl = lines!(ax, model_df_1.step, model_df_1.total_infected, color = :blue)
    recoveredl = lines!(ax, model_df_1.step, model_df_1.total_recovered, color = :orange)
    susceptiblel = lines!(ax, model_df_1.step, model_df_1.total_sus, color = :green)

    infectedl_2 = lines!(ax, model_df_2.step, model_df_2.total_infected, color = :red)
    recoveredl_2 = lines!(ax, model_df_2.step, model_df_2.total_recovered, color = :purple)
    susceptiblel_2 = lines!(ax, model_df_2.step, model_df_2.total_sus, color = :black)

    # infectedl_3 = lines!(ax, model_df_2.step, model_df_3.total_infected, color = :blue)
    # recoveredl_3 = lines!(ax, model_df_3.step, model_df_3.total_recovered, color = :orange)
    # susceptiblel_3 = lines!(ax, model_df_3.step, model_df_3.total_sus, color = :green)


    figure[1, 2] = Legend(figure, [infectedl, recoveredl, susceptiblel,infectedl_2, recoveredl_2, susceptiblel_2], ["Infected (NO)", "Recovered (NO)", "Susceptible (NO)","Infected (OP)", "Recovered (OP)", "Susceptible (OP)"])
    figure

end





function plot_population_timeseries(model_df)
    figure = Figure(resolution = (600, 400))
    # model_df_sub_set = model_df
    # model_df = model_df[setdiff(1:end, 12), :]
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Number infected")
    infectedl = lines!(ax, model_df.step, model_df.total_infected, color = :blue)
    recoveredl = lines!(ax, model_df.step, model_df.total_recovered, color = :orange)
    susceptiblel = lines!(ax, model_df.step, model_df.total_sus, color = :green)

    figure[1, 2] = Legend(figure, [infectedl, recoveredl, susceptiblel], ["Infected", "Recovered", "Susceptible"])
    figure
end

## Optimising function

# original function - not working
# function cost(x)
#     model = model_initialize(;
#         beta = x[1],
#         gamma = x[2],
#     )
#
#     infected_fraction(model) =
#         count(a.status == :I for a in allagents(model)) / nagents(model)
#
#     _, data = run!(model,dummystep,model_step!,20;
#                     mdata = [infected_fraction], when_model = [14],
#                     replicates = 10,)
#
#     return mean(data.infected_fraction)#data.total_infected, data.total_recovered, data.total_sus
# end

# new cost function
function myCost(x, ys)
    model = model_initialize(;
        beta = x[1],
        gamma = x[2],
    )

    _, data = run!(model,dummystep,model_step!,20;
                    mdata = [total_infected], when_model = collect(1:14),
                   replicates = 1,)
    s = 0
    for i = 1:14
        s = s + (data.total_infected[i] - ys[i])^2
    end
    return Float64(s)
end

#define part cost function
partCost(ys) = x -> myCost(x, ys)

Random.seed!(10)

#x0 with values used in simulation
x0 = [
    2,
    0.5,
]

#actual values
actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]

#cost(x0)

#part cost of actuals vs simulation values
partCost(actuals)(x0)

# result = bboptimize(cost,
#     SearchRange = [
#         (1, 10),
#         (0.1, 1),
#    ],
#    NumDimensions = 2,
#    MaxTime = 20,
#                     )

# optimise result
result = bboptimize(partCost(actuals),
    SearchRange = [
        (1, 10),
        (0.1, 1),
   ],
   NumDimensions = 2,
   MaxTime = 20,
                    )

 best_fitness(result)

 x = best_candidate(result)

 Random.seed!(0)

#model generated from optimised values
 model_1 = model_initialize(;
     beta = x[1],
     gamma = x[2],
 )

# # best senario where total_inf = 2, total_recovered = 760, total_sus = 1
#  _, data_1 = run!(model, dummystep, model_step!,20;  mdata = [nagents], when_model = [14],
#  replicates = 10,)
#
# mean(data_1.nagents)
#
# _, data_2 = run!(model, dummystep, model_step!,20;  mdata = [total_infected, total_recovered, total_sus], when_model = [14],
# replicates = 10,)
#
# data_2


#model_non_op = model_initialize(beta = 2, gamma = 0.5)
model_op = model_initialize(beta = x[1], gamma = x[2])

adata = [:status]

mdata = [total_infected, total_recovered, total_sus]

#_, model_df_1 = run!(model_non_op,dummystep,model_step!,10; adata, mdata)
_, model_df_2 = run!(model_op,dummystep,model_step!,14; adata, mdata);

#plot the one with optimal params by eye
model_1 = model_initialize(beta = 2.1, gamma = 0.85, seed = 50)

_, model_df_1 = run!(model_1,dummystep,model_step!,14; adata, mdata)

##new plots

#optimised version
#Plots.plot!(model_df_2.step, actuals[1:11], title = "Total population infected", labels = "Actual version", legend = :left, xlabel = "Day", ylabel = "Number of cases")

Gadfly.plot(layer(x = model_df_2.step, y = model_df_2.total_infected, Geom.line, Gadfly.Theme(default_color=color("red"))),
            layer(x = model_df_2.step, y = actuals, Geom.line, Gadfly.Theme(default_color=color("blue"))),
            layer(x = model_df_2.step, y = model_df_1.total_infected, Geom.line, Gadfly.Theme(default_color=color("green"))),
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







# model_2 = model_initialize(seed = 10)

 #mdata = [total_infected, total_recovered, total_sus]
 #cost(x0)

# ------------------
# Partcile Filtering
# ------------------

# # Parameter update covariace aka parameter diffusivity
# Q = [0.1 0.0; 0.0 0.01];
# # Observation covariance
# R = [0.1];
# # Number of particles
# N = 50;
# # S, I, beta and gamma
# nx = 4;
# # S and I since S + I + R = 763 always - no boys die
# ny = 2;
#
# function resample_stratified(weights)
#
#     N = length(weights)
#     # make N subdivisions, and chose a random position within each one
#     positions =  (rand(N) + collect(range(0, N - 1, length = N))) / N
#
#     indexes = zeros(Int64, N)
#     cumulative_sum = cumsum(weights)
#     i, j = 1, 1
#     while i <= N
#         if positions[i] < cumulative_sum[j]
#             indexes[i] = j
#             i += 1
#         else
#             j += 1
#         end
#     end
#     return indexes
# end
#
# function pf(inits, N, f, h, y, Q, R, nx, ny)
#     # inits - initial values
#
#     # N - number of particles
#
#     # f is the state update function - for us this is the ABM - takes
#     # a state and the parameters and returns a new state one time step
#     # forward
#
#     # h is the observation function - for us the state is S, I, R
#     # (although S + I + R = total boys) but we can only observe I
#
#     # y is the data - for us this the number infected for each of the
#     # 14 days
#
#     # To avoid particle collapse we need to perturb the parameters -
#     # we do this by sampling from a multivariate normal distribution
#     # with a given covariance matrix Q
#
#     # R is the observation covariance matrix
#
#     # nx is the dimension of the state - we have S I R beta and gamma
#     # but S + I + R is fixed so we can take the state to be S I beta
#     # gamma i.e. nx = 4
#
#     # The state update function (running the ABM) returns the new
#     # state so we need to know which of the nx state variables are the
#     # actual state and which are parameters which we now consider to
#     # be state. ny is the number of actual state variables so that
#     # nx - ny is the number of parameters.
#
#     T = length(y)
#     log_w = zeros(T,N);
#     x_pf = zeros(nx,N,T);
#     x_pf[:,:,1] = inits;
#     wn = zeros(N);
#
#     for t = 1:T
#         if t >= 2
#             a = resample_stratified(wn);
#
#             x_pf[1 : ny, :, t] = hcat(f(x_pf[:, a, t - 1])...)
#             x_pf[ny + 1 : ny + nx, :, t] = x_pf[ny + 1 : ny + nx, a, t - 1] + rand(MvNormal(zeros(nx - ny), Q), N)
#         end
#
#         log_w[t, :] = logpdf(MvNormal(y[t, :], R), h(x_pf[:,:,t]));
#
#         # To avoid underflow subtract the maximum before moving from
#         # log space
#
#         wn = map(x -> exp(x), log_w[t, :] .- maximum(log_w[t, :]));
#         wn = wn / sum(wn);
#     end
#
#     log_W = sum(map(log, map(x -> x / N, sum(map(exp, log_w[1:T, :]), dims=2))));
#
#     return(x_pf, log_w, log_W)
#
# end
