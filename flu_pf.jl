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
using StatsPlots

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
Q = [0.01 0.0; 0.0 0.005]; #--> spread is ok, peak is not
#Q = [0.01 0.0; 0.0 0.01]; #--> spread does not change, peak is flattened
#Q = [0.001 0.0; 0.0 0.001]; #--> spread effected, peak is flattened
#Q = [0.005 0.0; 0.0 0.05]; #--> does not reflect spread

# Observation covariance
R = Matrix{Float64}(undef,1,1);
R[1,1] = 0.1;
# Number of particles
N = 10000; #1000;
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


# end_states is a 5x1000x15 array --> so observe 5 variables, 1000 particles and 15 time steps
# so {:,:,1} = 762 762 = mean(row)
#              1.0 1.0 = mean(row) --> these means will then be the point for t = 1
(end_states, bar, baz) = pf(inits, N, simulate_one_step, observe, y, Q, R, nx, ny);

## Plots of mean sus, infected, recovered, beta and gamma and std --> based on time step

overall_mean_matrix = zeros(15,5)
overall_std_matrix = zeros(15,5)
overall_median_matrix = zeros(15,5)
#calculate mean based on time step and store in a overall matrix for plotting
for i in 1:15
    (INF, REC,SUS,BE,GA) = mean(end_states[:,:,i], dims = 2)
    overall_mean_matrix[i,1] = INF
    overall_mean_matrix[i,2] = REC
    overall_mean_matrix[i,3] = SUS
    overall_mean_matrix[i,4] = BE
    overall_mean_matrix[i,5] = GA
end

for i in 1:15
    (INF, REC,SUS,BE,GA) = std(end_states[:,:,i], dims = 2)
    overall_std_matrix[i,1] = INF
    overall_std_matrix[i,2] = REC
    overall_std_matrix[i,3] = SUS
    overall_std_matrix[i,4] = BE
    overall_std_matrix[i,5] = GA
end

for i in 1:15
    (INF, REC,SUS,BE,GA) = median(end_states[:,:,i], dims = 2)
    overall_median_matrix[i,1] = INF
    overall_median_matrix[i,2] = REC
    overall_median_matrix[i,3] = SUS
    overall_median_matrix[i,4] = BE
    overall_median_matrix[i,5] = GA
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
lines = [layer(x= step_vec,y= end_states[2,i,:], Geom.line,Gadfly.Theme(line_width = 0.6mm)) for i in range(1,stop=1000)] #change this for more paths
actuals_line = [layer(x = step_vec, y = actuals, Geom.line, Gadfly.Theme(default_color=color("red"),line_width = 1mm))]
two_plus_std_line = [layer(x = step_vec, y = 2*overall_std_matrix[:,2] + overall_mean_matrix[:,2]  , Geom.line, Gadfly.Theme(default_color=color("black"),line_width = 0.7mm))]
two_minus_std_line = [layer(x = step_vec, y = overall_mean_matrix[:,2] - 2*overall_std_matrix[:,2] , Geom.line, Gadfly.Theme(default_color=color("green"), line_width = 0.7mm))]
mean_line = [layer(x = step_vec, y = overall_mean_matrix[:,2], Geom.line, Gadfly.Theme(default_color=color("blue"), line_width = 0.8mm))]
median_line = [layer(x = step_vec, y = overall_median_matrix[:,2], Geom.line, Gadfly.Theme(default_color=color("purple"),line_width = 1.4mm))]


#append!(actuals_line,lines)
#append!(actuals_line, median_line)
append!(actuals_line,mean_line)
append!(actuals_line, median_line)
append!(actuals_line, two_plus_std_line)
append!(actuals_line,two_minus_std_line)
#append!(actuals_line, median_line)
append!(actuals_line,lines)

#all plots
Gadfly.plot(actuals_line...,Coord.Cartesian(ymin=-0.1,ymax=763), Guide.XLabel("Day"),Guide.YLabel("Population"), Guide.Title("Plot of infected individuals overtime"), Guide.manual_color_key("Legend", ["Paths of first 1000 particles","Actual data", "Mean data", "2+ std", "2- std", "Median data"], ["deepskyblue", "red", "blue", "black", "green", "purple"]))

# only paths of particles
Gadfly.plot(lines..., Guide.XLabel("Day"),Guide.YLabel("Population"),Guide.Title("Plot of infected individuals overtime"), Guide.manual_color_key("Legend", ["Paths of first 1000 particles"], ["deepskyblue"]))

Gadfly.plot(median_line...)
## Histograms
#variation for each param, at all time steps
col = Colors.distinguishable_colors(18)

##for overlaying plots
# inf_variation = [layer(x= end_states[1,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]
# inf_variation = [layer(x= end_states[2,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]
# rec_variation = [layer(x= end_states[3,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]
# beta_variation = [layer(x= end_states[4,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]
# gamma_variation = [layer(x= end_states[5,:,i], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]
#
# p1 = Gadfly.plot(sus_variation..., Guide.Title("Susceptible"))
# p2 = Gadfly.plot(inf_variation..., Guide.Title("Infected"))
# p3 = Gadfly.plot(rec_variation...,Guide.Title("Recovered"))
# p4 = Gadfly.plot(beta_variation..., Guide.Title("Beta"))
# p5 = Gadfly.plot(gamma_variation..., Guide.Title("Gamma"))
#
# gridstack(Union{Plot,Compose.Context}[p1 p2 p3; p4 p5 Compose.context()])

##for seperate plots
#limit to 650-800
#original
#sus_variation = [Gadfly.plot(x= end_states[1,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Population"), Gadfly.Theme(default_color=col[i])) for i in range(1,stop=15)]

sus_variation_1_4 = [Gadfly.plot(x= end_states[1,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Population"), Gadfly.Theme(default_color=col[6]), Coord.Cartesian(xmin=730,xmax=770)) for i in range(1,stop=4)]
sus_variation_5_6 = [Gadfly.plot(x= end_states[1,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Population"), Gadfly.Theme(default_color=col[6]), Coord.Cartesian(xmin=400,xmax=700)) for i in range(1,stop=6)]
sus_variation_7 = [Gadfly.plot(x= end_states[1,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Population"), Gadfly.Theme(default_color=col[6]), Coord.Cartesian(xmin=0,xmax=400)) for i in range(7,stop=7)]
sus_variation_8 = [Gadfly.plot(x= end_states[1,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Population"), Gadfly.Theme(default_color=col[6]), Coord.Cartesian(xmin=0,xmax=150)) for i in range(8,stop=8)]
sus_variation_9_15 = [Gadfly.plot(x= end_states[1,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Population"), Gadfly.Theme(default_color=col[6]), Coord.Cartesian(xmin=0,xmax=20)) for i in range(9,stop=15)]


inf_variation = [Gadfly.plot(x= end_states[2,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Population"), Gadfly.Theme(default_color=col[7])) for i in range(1,stop=15)]
rec_variation = [Gadfly.plot(x= end_states[3,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Population"), Gadfly.Theme(default_color=col[8])) for i in range(1,stop=15)]
beta_variation = [Gadfly.plot(x= end_states[4,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Beta"),Coord.Cartesian(xmin=1,xmax=4.5), Gadfly.Theme(default_color=col[12])) for i in range(1,stop=15)]
gamma_variation = [Gadfly.plot(x= end_states[5,:,i], Geom.histogram(bincount = 10),Guide.XLabel("Gamma"),Coord.Cartesian(xmin=0.3,xmax=1), Gadfly.Theme(default_color=col[10])) for i in range(1,stop=15)]


gamma_v = [layer(x= end_states[5,:,i], Geom.violin,Gadfly.Theme(default_color=col[10])) for i in range(1,stop=15)]

#Gadfly.plot(y=end_states[5,:,1], Geom.violin)
# p12 = Gadfly.plot(beta_variation[1])

Gadfly.plot(gamma_v...)


#Gadfly.plot(x=step_vec, y=end_states[5,:,i], Geom.violin)


# gamma_v1 = [layer(x= step_vec, y = end_states[5,:,i], Geom.violin,Gadfly.Theme(default_color=col[10])) for i in range(1,stop=15)]
#
# step_vec_1 = [1];
# step_vec_2 = [2];


# Gadfly.plot(#layer(x = step_vec_1 , y = end_states[5,:,1], Geom.violin),
#             layer(x = step_vec_2, y = end_states[5,:,2], Geom.violin))
#
# StatsPlots.violin!(y= end_states[5,:,1])
#


group = vcat(fill("1", 10000),fill("2", 10000),fill("3", 10000),fill("4", 10000),fill("5", 10000),fill("6", 10000), fill("7", 10000),fill("8", 10000),fill("9", 10000), fill("10", 10000),
            fill("11", 10000), fill("12", 10000),
            fill("13", 10000), fill("14", 10000),
            fill("15", 10000))


group_1 = vcat(fill("1", 10000))

group_2 = vcat(fill("2", 10000),fill("3", 10000),fill("4", 10000),fill("5", 10000),fill("6", 10000), fill("7", 10000),fill("8", 10000),fill("9", 10000), fill("10", 10000),
            fill("11", 10000), fill("12", 10000),
            fill("13", 10000), fill("14", 10000),
            fill("15", 10000))

#beta
data_beta_1 = vcat(end_states[4,:,1])

data_beta_2 = vcat(end_states[4,:,2],end_states[4,:,3],
                    end_states[4,:,4],end_states[4,:,5],end_states[4,:,6],
                    end_states[4,:,7],end_states[4,:,8],end_states[4,:,9],
                    end_states[4,:,10],end_states[4,:,11], end_states[4,:,12],
                    end_states[4,:,13], end_states[4,:,14],end_states[4,:,15])




data_gamma_1 = vcat(end_states[5,:,1])

data_gamma_2 = vcat(end_states[5,:,2],end_states[5,:,3],end_states[5,:,4],end_states[5,:,5],end_states[5,:,6],end_states[5,:,7], end_states[5,:,8],end_states[5,:,9], end_states[5,:,10],
            end_states[5,:,11], end_states[5,:,12],
            end_states[5,:,13], end_states[5,:,14],
            end_states[5,:,15])



df_gamma_1 = DataFrame(data=data_gamma_1, group=group_1)
df_gamma_2 = DataFrame(data=data_gamma_2, group=group_2)
df_beta_1 = DataFrame(data=data_beta_1, group=group_1)
df_beta_2 = DataFrame(data=data_beta_2, group=group_2)

g1 = Gadfly.plot(df_gamma_1, x=:group, y=:data, Geom.violin,Guide.XLabel("Time step"), Guide.YLabel("Gamma"))
g2 = Gadfly.plot(df_gamma_2, x=:group, y=:data, Geom.violin,Guide.XLabel("Time step"), Guide.YLabel("Gamma"))
b1 = Gadfly.plot(df_beta_1, x=:group, y=:data, Geom.violin,Guide.XLabel("Time step"), Guide.YLabel("Beta"))
b2 = Gadfly.plot(df_beta_2, x=:group, y=:data, Geom.violin,Guide.XLabel("Time step"), Guide.YLabel("Beta"))


Gadfly.title(hstack(g1,g2), "Variation in Gamma")
Gadfly.title(hstack(b1,b2), "Variation in Beta")





# gridstack([])
#

# for i in 2:15
#     initial = end_states[5,:,1]
#     new = end_states[5,:,i]
#
#
# end
#
#
#
# d = (vcat(end_states[5,:,i]) for i in range(1,stop=15)
# gridstack(reshape([[render(sus_variation[i]) for i in 1:15], Gtk.canvas()],3,5))

##actual plots
#susceptible
Gadfly.title(vstack(hstack( sus_variation_1_4[1], sus_variation_1_4[2], sus_variation_1_4[3]),
                    hstack(sus_variation_1_4[4], sus_variation_5_6[5], sus_variation_5_6[6]))
                    ,"Variation in susceptible (time steps 1-6)")


Gadfly.title(vstack(hstack( sus_variation_7[1], sus_variation_8[1], sus_variation_9_15[1]),
                    hstack(sus_variation_9_15[2], sus_variation_9_15[3], sus_variation_9_15[4]))
                    ,"Variation in susceptible (time steps 7-12)")

Gadfly.title(vstack(hstack(sus_variation_9_15[5], sus_variation_9_15[6], sus_variation_9_15[7]),
              hstack())
              ,"Variation in susceptible (time steps 13-15)")



 # infected
Gadfly.title(vstack(hstack(inf_variation[1], inf_variation[2], inf_variation[3]),
                    hstack(inf_variation[4], inf_variation[5], inf_variation[6]))
                    ,"Variation in infected(time steps 1-6)")

Gadfly.title(vstack(hstack(inf_variation[7], inf_variation[8], inf_variation[9]),
                    hstack(inf_variation[10], inf_variation[11], inf_variation[12]))
                    ,"Variation in infected(time steps 7-12)")

Gadfly.title(vstack(hstack(inf_variation[13], inf_variation[14], inf_variation[15]),
                    hstack())
                    ,"Variation in infected(time steps 13-15)")


 #recovered
Gadfly.title(vstack(hstack(rec_variation[1], rec_variation[2], rec_variation[3]),
                    hstack(rec_variation[4], rec_variation[5], rec_variation[6]))
                    ,"Variation in recovered(time steps 1-6)")

Gadfly.title(vstack(hstack(rec_variation[7], rec_variation[8], rec_variation[9]),
                     hstack(rec_variation[10], rec_variation[11], rec_variation[12]))
                    ,"Variation in recovered(time steps 7-12)")

Gadfly.title(vstack(hstack(rec_variation[13], rec_variation[14], rec_variation[15]),
                    hstack())
                    ,"Variation in recovered(time steps 13-15)")



#Beta
Gadfly.title(vstack(hstack(beta_variation[1], beta_variation[2], beta_variation[3]),
                    hstack(beta_variation[4], beta_variation[5], beta_variation[6]))
                    ,"Variation in Beta (time steps 1-6)")

Gadfly.title(vstack(hstack(beta_variation[7], beta_variation[8], beta_variation[9]),
                    hstack(beta_variation[10], beta_variation[11], beta_variation[12]))
                    ,"Variation in Beta (time steps 7-12)")

Gadfly.title(vstack(hstack(beta_variation[13], beta_variation[14], beta_variation[15]),
                    hstack())
                    ,"Variation in Beta (time steps 13-15)")

#Gamma
Gadfly.title(vstack(hstack(gamma_variation[1], gamma_variation[2], gamma_variation[3]),
                    hstack(gamma_variation[4], gamma_variation[5], gamma_variation[6]))
                    ,"Variation in Gamma (time steps 1-6)")

Gadfly.title(vstack(hstack(gamma_variation[7], gamma_variation[8], gamma_variation[9]),
                    hstack(gamma_variation[10], gamma_variation[11], gamma_variation[12]))
                    ,"Variation in Gamma (time steps 7-12)")

Gadfly.title(vstack(hstack(gamma_variation[13], gamma_variation[14], gamma_variation[15]),
                    hstack())
                    ,"Variation in Gamma (time steps 13-15)")





##Plot layout options

# Gadfly.title(vstack(hstack( sus_variation_1_4[1], sus_variation_1_4[2], sus_variation_1_4[3]),
#                     hstack(sus_variation_1_4[4], sus_variation_5_6[5], sus_variation_5_6[6]))
#                     ,"Variation in susceptible (time steps 1-6)")
# #susceptible
# Gadfly.title(vstack(hstack(sus_variation[1], sus_variation[2], sus_variation[3]),
#                     hstack(sus_variation[4], sus_variation[5], sus_variation[6]))
#                     ,"Variation in susceptible (time steps 1-6)")
#
# Gadfly.title(vstack(hstack(sus_variation[7], sus_variation[8], sus_variation[9]),
#                     hstack(sus_variation[10], sus_variation[11], sus_variation[12]))
#                     ,"Variation in susceptible (time steps 7-12)")
#
# Gadfly.title(vstack(hstack(sus_variation[13], sus_variation[14], sus_variation[15]),
#                     hstack())
#                     ,"Variation in susceptible (time steps 13-15)")
#
#
#
#
#
#
# #3x5
# gridstack([sus_variation[1] sus_variation[2] sus_variation[3];
#            sus_variation[4] sus_variation[5] sus_variation[6];
#            sus_variation[7] sus_variation[8] sus_variation[9];
#            sus_variation[10] sus_variation[11] sus_variation[12];
#            sus_variation[13] sus_variation[14] sus_variation[15];])
# #4x4
# gridstack(Union{Plot,Compose.Context}[sus_variation[1] sus_variation[2] sus_variation[3] sus_variation[4];
#            sus_variation[5] sus_variation[6] sus_variation[7] sus_variation[8];
#            sus_variation[9] sus_variation[10] sus_variation[11] sus_variation[12];
#            sus_variation[13] sus_variation[14] sus_variation[15] Compose.context();])
#
# #2x4x2
# Gadfly.title(vstack(hstack(sus_variation[1], sus_variation[2], sus_variation[3], sus_variation[4]),
#                     hstack(sus_variation[5], sus_variation[6], sus_variation[7], sus_variation[8]))
#                     ,"Variation in susceptible (time steps 1-8)")
#
# Gadfly.title(vstack(hstack(sus_variation[9], sus_variation[10], sus_variation[11], sus_variation[12]),
#                     hstack(sus_variation[13], sus_variation[14], sus_variation[15]))
#                     ,"Variation in susceptible (time steps 9-15)")
#
#
#
# ## Plots for other params
# # Infected
# Gadfly.title(vstack(hstack(inf_variation[1], inf_variation[2], inf_variation[3], inf_variation[4]),
#                     hstack(inf_variation[5], inf_variation[6], inf_variation[7], inf_variation[8]))
#                     ,"Variation in infected(time steps 1-8)")
#
# Gadfly.title(vstack(hstack(inf_variation[9], inf_variation[10], inf_variation[11], inf_variation[12]),
#                     hstack(inf_variation[13], inf_variation[14], inf_variation[15]))
#                     ,"Variation in infected (time steps 9-15)")
#
# # Recovered
# Gadfly.title(vstack(hstack(rec_variation[1], rec_variation[2], rec_variation[3], rec_variation[4]),
#                     hstack(rec_variation[5], rec_variation[6], rec_variation[7], rec_variation[8]))
#                     ,"Variation in recovered (time steps 1-8)")
#
# Gadfly.title(vstack(hstack(rec_variation[9], rec_variation[10], rec_variation[11], rec_variation[12]),
#                     hstack(rec_variation[13], rec_variation[14], rec_variation[15]))
#                     ,"Variation in recovered (time steps 9-15)")
#
#
# # Beta
# Gadfly.title(vstack(hstack(beta_variation[1], beta_variation[2], beta_variation[3], beta_variation[4]),
#                     hstack(beta_variation[5], beta_variation[6], beta_variation[7], beta_variation[8]))
#                     ,"Variation in Beta (time steps 1-8)")
#
# Gadfly.title(vstack(hstack(beta_variation[9], beta_variation[10], beta_variation[11], beta_variation[12]),
#                     hstack(beta_variation[13], beta_variation[14], beta_variation[15]))
#                     ,"Variation in Beta (time steps 9-15)")
#
#
# # Gamma
# Gadfly.title(vstack(hstack(gamma_variation[1], gamma_variation[2], gamma_variation[3], gamma_variation[4]),
#                     hstack(gamma_variation[5], gamma_variation[6], gamma_variation[7], gamma_variation[8]))
#                     ,"Variation in Gamma (time steps 1-8)")
#
# Gadfly.title(vstack(hstack(gamma_variation[9], gamma_variation[10], gamma_variation[11], gamma_variation[12]),
#                     hstack(gamma_variation[13], gamma_variation[14], gamma_variation[15]))
#                     ,"Variation in Gamma (time steps 9-15)")
#
#
#
#
#
#
# # vstack(hstack(p1,p2), p3)
# # vstack(p1,p2)
# # vstack(p4,p5)
# # #hstack(p1, p2, p3)
# # #gridstack([p4 ; p5])
#
#
#
#
#
#
# #p1 = Gadfly.plot(x = end_states[4,:,1], Geom.histogram(bincount = 10), Gadfly.Theme(default_color=color("black")))
#
#
#


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
