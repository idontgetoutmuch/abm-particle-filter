#Koalas with space
using Agents, Random
using InteractiveDynamics
using CairoMakie
using Plots
using Statistics


#Define koala/Eucalyptus agent struct
mutable struct KoalaOrEucalyptus <: AbstractAgent
    id::Int #agent id
    pos::NTuple{2,Int} #agent's position on GridSpace
    type::Symbol #:koala or :Eucalyptus
    death_prob::Float64 #only for Koala
    production_rate::Float64 #only for eucalyptus
    consume_rate::Float64 #only for eucalyptus
end

#define Koala and Eucalyptus agents for easier use
Koala(id,pos,death_prob,production_rate, consume_rate) = KoalaOrEucalyptus(id, pos, :koala, death_prob, production_rate, consume_rate)
Eucalyptus(id, pos, death_prob,production_rate, consume_rate) = KoalaOrEucalyptus(id, pos, :eucalyptus, death_prob, production_rate, consume_rate)


#initalise model with space and agents
function initialize_model(;
    n_koala = 10, #initial koala population
    n_eucalyptus = 20, #initial eucalyptus population
    input_params:: Vector{Float64},
    seed = 23182, #seed for model
     )
     koala_death_rate = input_params[1] #death rate of koala agents
     eucalyptus_production_rate = input_params[2]#prduction rate of eucalyptus
     eucalyptus_consume_rate = input_params[3] #death/consumption rate of eucalyptus
     space = GridSpace((20, 20); periodic = true) #initalise grid space with periodic boundary conditions
     #Define properties of model for easier referencing
     properties = Dict(:euca_pr => eucalyptus_production_rate, :koala_dr => koala_death_rate, :euca_cr => eucalyptus_consume_rate)

     rng = MersenneTwister(seed) #rng for model
     #define the Koalas ABM with the propeties, space and rng
     model = ABM(KoalaOrEucalyptus, space; properties,rng, scheduler = Schedulers.randomly)

     id = 0; #initialise for agents

     #add initial number of koalas to model
     for _ in 1:n_koala
         id += 1
         koala = Koala(id,(1, 1),koala_death_rate, 0,0) #define Koala agent with approriate properties
         add_agent_single!(koala, model) #add agent to unoccupied position
     end
     #add initial number of eucalyptus to model
     for _ in 1:n_eucalyptus
         id += 1
         #define Eucalyptus agent with approriate properties
         eucalyptus = Eucalyptus(id, (1, 1), 0, eucalyptus_production_rate, eucalyptus_consume_rate)
         add_agent_single!(eucalyptus, model) #add agent to unoccupied position
     end

     return model #return the initalised model
 end

#helper functions to count amount koala and eucalyptus agents
leaf(m) = count(a.type == :eucalyptus for a in allagents(m))
koalas(m) = count(a.type == :koala for a in allagents(m))

#agent step that executes for every time step
 function agent_step!(agent::KoalaOrEucalyptus,model)
     #if the current agent is a koala, then execute the koala step function, else
     # execute the eucalytpus step
     if agent.type == :koala
         koala_step!(agent,model)
    else
        eucalyptus_step!(agent,model)
    end

    #current number of koala and eucalyptus agents
    num_euca = leaf(model)
    num_koalas = koalas(model)

    #if there are no eucalytpus agents in simulation, then
    #based on the production rate, add an eucalyptus agent to sim
    if num_euca == 0
        if rand() < model.euca_pr
            id = nextid(model)
            eucalyptus_new = Eucalyptus(id, (1, 1), 0, model.euca_pr, model.euca_cr)
            add_agent_single!(eucalyptus_new, model) #add agent to unoccupied position
        end
    end

    #if there are no koala agents in simulation, then
    #based on the eucalyptus consumption rate, add a koala agent to sim
    if num_koalas == 0
        if rand() < model.euca_cr
            id = nextid(model)
            k_new = Koala(id,(1, 1), model.koala_dr, 0,0)
            add_agent_single!(k_new, model)
        end
    end



 end

#Defines the behaviour of koala agents
 function koala_step!(koala,model)
     #decide the agent's movement on grid based on random walk
     upper_bound = 0; #right and left bound
     lower_bound = 0; #up and down bound
     prob_walk = rand() #generate a random number
     if prob_walk < 0.25 #north so move up 1 and down 0
         upper_bound = 0; #no left or right movement
         lower_bound = 1; #move up 1
    elseif prob_walk >= 0.25 && prob_walk < 0.5 #south, move down 1
        upper_bound = 0;
        lower_bound = -1;
    elseif prob_walk >= 0.5 && prob_walk < 0.75 #east, move right 1
        upper_bound = 1;
        lower_bound = 0;
    else
        upper_bound = -1; #west, move left 1
        lower_bound = 0;
    end

    #move agent based on bounds  -> ifempty may need to be false
     walk!(koala, (upper_bound,lower_bound), model; ifempty = true)


     num_koalas = koalas(model) #current num koalas
     num_eucalyptus = leaf(model) # current num of eucalyptus

     #to ensure only one event occurs
     event_prob = rand()
     if event_prob < koala.death_prob
          kill_agent!(koala,model)
     else
          if rand() < model.euca_cr
              for neighbor in nearby_agents(koala, model)
                  if neighbor.type == :eucalyptus #if neighbour is eucalyptus
                      kill_agent!(neighbor,model) #the eucalyptus gets eaten
                      id = nextid(model)
                      koala_new = Koala(id,(1, 1), model.koala_dr, 0,0)
                      add_agent_single!(koala_new, model) #new koala gets added to model
                  break #breaks after 1 iteration (only consume 1)
              end
          end
      end
  

    end

     #after, chance of killng  the agent based on prob

end

#Defines the behaviour of eucalyptus agents
function eucalyptus_step!(eucalyptus,model)
    #agent has no spatial movements
    new_euca = leaf(model) #calculate num of eucalyptus
    if rand() < eucalyptus.production_rate*(1-(new_euca)/1000)
                #new eucalyptus if the above condition is met
                id = nextid(model)
                eucalyptus_new = Eucalyptus(id, (1, 1), 0, model.euca_pr, model.euca_cr)
                add_agent_single!(eucalyptus_new, model)
    end
end


 koala(a) = a.type == :koala
 eucalyptus(a) = a.type == :eucalyptus

 koala_death_rate = 0.001 #death rate of koala agents
 eucalyptus_production_rate = 0.08 #production rate of eucalyptus
 eucalyptus_consume_rate = 0.008 #death/consumption rate of eucalyptus

 ABM_params = [koala_death_rate;eucalyptus_production_rate;eucalyptus_consume_rate]

#running model with 2000 time steps
 model = initialize_model(input_params = ABM_params)
 n_steps = 2000
 adata = [(koala, count), (eucalyptus, count)]
 adf, _ = run!(model, agent_step!, dummystep, n_steps; adata)

#function to plot one simulation of model
 function plot_population_timeseries(adf)
     figure = Figure(resolution = (600, 400))
     ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
     koalal = lines!(ax, adf.step, adf.count_koala, color = :blue)
     eucalyptusl = lines!(ax, adf.step, adf.count_eucalyptus, color = :green)
     figure[1, 2] = Legend(figure, [koalal, eucalyptusl], ["Koalas", "Eucalyptus"])
     figure
 end

 plot_population_timeseries(adf)

# ## Running 1000 simulations + plots

 a_koalas = adf[:,2];
 a_euca = adf[:,3];

 for i in 2:100
     model_2 = initialize_model(input_params = ABM_params)
     n_step = 2000;
     agent_df_2, _ = run!(model_2, agent_step!, dummystep,n_step; adata);
     a_1 = agent_df_2[:,2];
     a_2 = agent_df_2[:,3];
     global a_koalas = [a_koalas a_1]
     global a_euca = [a_euca a_2]
     print("#####################################################################\n");
     print(" Iteration: ", i, " completed.\n");
 end

 #Plots.histogram(a_euca[2001,:], bins = 20)

 mean_of_50 = mean(a_koalas, dims = 2)
 median_of_50 = median(a_koalas, dims = 2)
 std_of_50 = std(a_koalas, dims = 2)

 mean_of_50_e = mean(a_euca, dims = 2)
 median_of_50_e = median(a_euca, dims = 2)
 std_of_50_e = std(a_euca, dims = 2)


#all plots
 Plots.plot(adf.step, a_koalas[:,1],  label = "", title = "ABM populations from 100 runs",xlab="Time step",ylabel="Number", linecolor = :red, titlefontsize = 8)
 Plots.plot!(adf.step, a_euca[:,1], label = "",xlab="Time step",ylabel="Number", linecolor = :blue)
 for i in 2:(size(a_koalas)[2]-1)
     #Plots.plot!(t, a_50[:,i])#, seriestype = :scatter)
     Plots.plot!(adf.step, a_koalas[:,i], linecolor = :red, label = "")
     Plots.plot!(adf.step, a_euca[:,i], linecolor = :blue, label = "")

 end
 #Plots.plot!(t,a_50[:,50])#, seriestype = :scatter)
 Plots.plot!(adf.step, a_koalas[:,size(a_koalas)[2]], label = "Koala", linecolor = :red)
 Plots.plot!(adf.step, a_euca[:,size(a_koalas)[2]],label = "Eucalyptus", linecolor = :blue, legend = :right, legendfontsize = 6)
#
 Plots.plot!(adf.step, mean_of_50, label = "Mean (Koalas)", lw = 3, linecolor = :black)
 Plots.plot!(adf.step, 2*std_of_50 + mean_of_50, label = "2+ std (Koalas)", lw = 3, linecolor = :black, linestyle = :dash )
 Plots.plot!(adf.step, abs.(mean_of_50 - 2*std_of_50) , label = "2- std(Koalas)", lw = 3.5, linecolor = :black, linestyle = :dot )

 Plots.plot!(adf.step, mean_of_50_e, label = "Mean (Euca.)", lw = 3, linecolor = :green1)
 Plots.plot!(adf.step, 2*std_of_50_e + mean_of_50_e, label = "2+ std (Euca.)", lw = 3,linecolor = :green1, linestyle = :dash)
 Plots.plot!(adf.step, abs.(mean_of_50_e - 2*std_of_50_e) , label = "2- std(Euca.)", lw = 3.5,linecolor = :green1, linestyle = :dot)





# # simulation
# model12 = initialize_model(input_params = ABM_params)
# offset(a) = a.type == :koala ? (-0.7, -0.5) : (-0.3, -0.5)
# ashape(a) = a.type == :koala ? :circle : :utriangle
# acolor(a) = a.type == :koala ? RGBAf0(0.2, 0.2, 0.2, 0.8) : RGBAf0(0.22, 0.58, 0.2, 0.8)

# plotkwargs = (
#     ac = acolor,
#     as = 15,
#     am = ashape,
#     offset = offset,
# )

# fig, _ = abm_plot(model12; plotkwargs...)
# fig


# # n_steps = 2000
# # adata = [(koala, count), (eucalyptus, count)]
# # adf2, _ = run!(model1, agent_step!, dummystep, n_steps; adata)
# #
# # plot_population_timeseries(adf2)

# abm_video(
#     "newKoala.mp4",
#     model12,
#     agent_step!;
#     frames = 200,
#     framerate = 8,
#     plotkwargs...,
# )


## average behaviour

# mean_of_50 = mean(a_koalas, dims = 2)
# median_of_50 = median(a_koalas, dims = 2)
# std_of_50 = std(a_koalas, dims = 2)
#
# mean_of_50_e = mean(a_euca, dims = 2)
# median_of_50_e = median(a_euca, dims = 2)
# std_of_50_e = std(a_euca, dims = 2)
#
# Plots.plot(adf.step, mean_of_50, label = "Mean", lw = 2,title = "Koala results from 1000 runs", xlab="Time",ylabel="Number",legend = :outertopright)
# Plots.plot!(adf.step, median_of_50, label = "Median", lw = 2)
# Plots.plot!(adf.step, 2*std_of_50 + mean_of_50, label = "2+ std", lw = 2)
# Plots.plot!(adf.step, abs.(mean_of_50 - 2*std_of_50) , label = "2- std", lw = 2)
# Plots.plot(adf.step, mean_of_50_e, label = "Mean", lw = 2,title = "Eucalyptus results from 1000 runs", xlab="Time",ylabel="Number",legend = :outertopright)
# Plots.plot!(adf.step, median_of_50_e, label = "Median", lw = 2)
# Plots.plot!(adf.step, 2*std_of_50_e + mean_of_50_e, label = "2+ std", lw = 2)
# Plots.plot!(adf.step, abs.(mean_of_50_e - 2*std_of_50_e) , label = "2- std", lw = 2)

##

# leftplot = subplot(1,2,1)
# Plots.plot(adf.step, mean_of_50, label = "Mean", lw = 2,title = "Results from 150 runs", xlab="Time",ylabel="Number",legend = :best)
# Plots.plot!(adf.step, median_of_50, label = "Median", lw = 2)
# Plots.plot!(adf.step, 2*std_of_50 + mean_of_50, label = "2+ std", lw = 2)
# Plots.plot!(adf.step, abs.(mean_of_50 - 2*std_of_50) , label = "2- std", lw = 2)
# rightplot = subplot(1,2,2)
# Plots.plot(adf.step, mean_of_50_e, label = "Mean", lw = 2, linestyle = :dash,title = "Eucalyptus results from 150 runs", xlab="Time",ylabel="Number",legend = :topleft)
# Plots.plot!(adf.step, median_of_50_e, label = "Median", lw = 2, linestyle = :dash)
# Plots.plot!(adf.step, 2*std_of_50_e + mean_of_50_e, label = "2+ std", lw = 2, linestyle = :dash)
# Plots.plot!(adf.step, abs.(mean_of_50_e - 2*std_of_50_e) , label = "2- std", lw = 2, linestyle = :dash)
#
# fig = gcf()
# title!(fig, "Nice plot")         # Same as `title("Nice plot")`
# title!(leftplot, "First plot")  # Set the title of the first plot
# fig   # The function `title!` does not update the visualization
