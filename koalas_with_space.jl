#Koalas with space
using Agents, Random
using InteractiveDynamics
using CairoMakie

mutable struct KoalaOrEucalyptus <: AbstractAgent
    id::Int
    pos::NTuple{2,Int}
    type::Symbol #:koala or :Eucalyptus
    death_prob::Float64 #only for Koala
    production_rate::Float64 #only for eucalyptus
    consume_rate::Float64 #only for eucalyptus
    #reproduction prob for later
end

Koala(id,pos,death_prob,production_rate, consume_rate) = KoalaOrEucalyptus(id, pos, :koala, death_prob, production_rate, consume_rate)
Eucalyptus(id, pos, death_prob,production_rate, consume_rate) = KoalaOrEucalyptus(id, pos, :eucalyptus, death_prob, production_rate, consume_rate)



function initialize_model(;
    n_koala = 5,
    n_eucalyptus = 20,
    koala_death_rate = 8e-3,
    eucalyptus_production_rate = 0.009,
    eucalyptus_consume_rate = 0.05,
    seed = 23182,
     )
     space = GridSpace((10, 10); periodic = false)
     properties = Dict(:euca_pr => eucalyptus_production_rate, :koala_dr => koala_death_rate, :euca_cr => eucalyptus_consume_rate)
     #properties = ()

     rng = MersenneTwister(seed)
     model = ABM(KoalaOrEucalyptus, space; properties,rng, scheduler = Schedulers.randomly)

     id = 0;

     for _ in 1:n_koala
         id += 1
         koala = Koala(id,(1, 1),koala_death_rate, 0,0)
         add_agent_single!(koala, model) #add agent to unoccupied position
         #add_agent!(koala, model)
     end

     for _ in 1:n_eucalyptus
         id += 1
         eucalyptus = Eucalyptus(id, (1, 1), 0, eucalyptus_production_rate, eucalyptus_consume_rate)
         add_agent_single!(eucalyptus, model)
         #add_agent!(eucalyptus, model)
     end

     return model
 end

leaf(m) = count(a.type == :eucalyptus for a in allagents(m))
koalas(m) = count(a.type == :koala for a in allagents(m))


 function agent_step!(agent::KoalaOrEucalyptus,model)
     # num_euc = leaf(model)
     # if num_euc < 1
     #     eucalyptus_step!(agent,model)
     if agent.type == :koala
         koala_step!(agent,model)
    else
        eucalyptus_step!(agent,model)
    end
 end

 function koala_step!(koala,model)
     #decide the agent's movement based on prob
     upper_bound = 0; #right and left bound
     lower_bound = 0; #up and down bound
     prob_walk = rand()
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

     # search nearby for eucalyptus -> nearby agents
     # then based on prob, consume it and then kill that agent
     #make new koala at random space
     num_koalas = koalas(model) #num koalas
     num_eucalyptus = leaf(model) #num of eucalyptus
     prob_contact = num_koalas/(num_eucalyptus + num_koalas) #determine prob of contact
     if rand() < prob_contact
         if rand() < model.euca_cr
             for neighbor in nearby_agents(koala, model)
                 if neighbor.type == :eucalyptus #if neighbour is eucalyptus
                     kill_agent!(neighbor,model) #the eucalyptus gets eaten
                     id = nextid(model)
                     koala_new = Koala(id,(1, 1), model.koala_dr, 0,0)
                     add_agent_single!(koala_new, model) #add agent to unoccupied position #new koala gets added to model
                     break #breaks after 1 iteration (only consume 1)
                 end
             end
         end
     end

     #after, we can also kill the agent based on prob
     if rand() < koala.death_prob
         kill_agent!(koala,model)
     end

 end

function eucalyptus_step!(eucalyptus,model)
    #can't move so never moves
    #add agent in position and then put in random for now
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

 model = initialize_model()
 n_steps = 1000
 adata = [(koala, count), (eucalyptus, count)]
 adf, _ = run!(model, agent_step!, dummystep, n_steps; adata)

 function plot_population_timeseries(adf)
     figure = Figure(resolution = (600, 400))
     ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
     koalal = lines!(ax, adf.step, adf.count_koala, color = :blue)
     eucalyptusl = lines!(ax, adf.step, adf.count_eucalyptus, color = :green)
     figure[1, 2] = Legend(figure, [koalal, eucalyptusl], ["Koalas", "Eucalyptus"])
     figure
 end

 plot_population_timeseries(adf)
