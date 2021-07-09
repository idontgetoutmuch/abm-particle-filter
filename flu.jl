using Agents, Random, DataFrames, LightGraphs
using Distributions: Poisson, DiscreteNonParametric
using DrWatson: @dict
using Plots
using Random
using InteractiveDynamics
using CairoMakie




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

## data collection and run model

#create the model
# model = model_initialize()
#
# # #models with different seeds
model_1 = model_initialize(seed = 125)
model_2 = model_initialize(seed = 10)
model_3 = model_initialize(seed = 500)


adata = [:status]

mdata = [total_infected, total_recovered, total_sus]

_, model_df_1 = run!(model_1,dummystep,model_step!,10; adata, mdata)
_, model_df_2 = run!(model_2,dummystep,model_step!,10; adata, mdata)
_, model_df_3 = run!(model_3,dummystep,model_step!,10; adata, mdata)


  #parameters = Dict(:beta => collect(2:3), :gamma => [0.5,0.6],)

# parameters = Dict(:seed => [10,500])
# # #
#   _, model_df = paramscan(parameters, model_initialize; adata, mdata, model_step!, n = 10)

##plotting function



# function plot_population_timeseries_altering(model_df, min,max)
#     figure = Figure(resolution = (600, 400))
#     ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Number infected")
#
#     infectedl = lines!(ax, model_df.step, model_df.total_infected, color = :blue)
#     recoveredl = lines!(ax, model_df.step, model_df.total_recovered, color = :orange)
#     susceptiblel = lines!(ax, model_df.step, model_df.total_sus, color = :green)
#
#     figure[1, 2] = Legend(figure, [infectedl, recoveredl, susceptiblel], ["Infected", "Recovered", "Susceptible"])
#     figure
# end
#
# function determine_lines(model_df)
#     figure = Figure(resolution = (600, 400))
#     ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Number infected")
#     infectedl = lines!(ax, model_df.step, model_df.total_infected, color = :blue)
#     recoveredl = lines!(ax, model_df.step, model_df.total_recovered, color = :orange)
#     susceptiblel = lines!(ax, model_df.step, model_df.total_sus, color = :green)
#
#     return infectedl, recoveredl, susceptiblel
# end

function plot_timeseries()
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Number infected")
    infectedl = lines!(ax, model_df_1.step, model_df_1.total_infected, color = :blue)
    recoveredl = lines!(ax, model_df_1.step, model_df_1.total_recovered, color = :orange)
    susceptiblel = lines!(ax, model_df_1.step, model_df_1.total_sus, color = :green)

    infectedl_2 = lines!(ax, model_df_2.step, model_df_2.total_infected, color = :blue)
    recoveredl_2 = lines!(ax, model_df_2.step, model_df_2.total_recovered, color = :orange)
    susceptiblel_2 = lines!(ax, model_df_2.step, model_df_2.total_sus, color = :green)

    infectedl_3 = lines!(ax, model_df_2.step, model_df_3.total_infected, color = :blue)
    recoveredl_3 = lines!(ax, model_df_3.step, model_df_3.total_recovered, color = :orange)
    susceptiblel_3 = lines!(ax, model_df_3.step, model_df_3.total_sus, color = :green)


    figure[1, 2] = Legend(figure, [infectedl, recoveredl, susceptiblel,infectedl_2, recoveredl_2, susceptiblel_2,infectedl_3, recoveredl_3, susceptiblel_3], ["Infected", "Recovered", "Susceptible","Infected", "Recovered", "Susceptible","Infected", "Recovered", "Susceptible"])
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


plot_timeseries()



    #
    # Plots.plot(model_df_1.step, model_df_1.total_infected, label = "Infected", color="blue")
    # Plots.plot!(model_df_2.step, model_df_2.total_infected, label = "")

    #
    # Plots.plot(model_df_1.step, model_df_1.total_infected)
    # Plots.plot!(model_df_2.step, model_df_2.total_infected)

# function subset_data(model_df)
#     for i in 1:size(model_df, 1)
#         if (model_df.step)
#
#     end
#
#
#     A = DataFrame()
#     while (model_df.step )
#
#
#
#     for i in model_df
#         while
#
#     end
# end


#
# plot_population_timeseries(model_df_1)
# plot_population_timeseries(model_df_2)
# plot_population_timeseries(model_df_3)
