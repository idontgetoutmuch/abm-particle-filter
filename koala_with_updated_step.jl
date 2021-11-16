#new Koalas
using Agents, Random
using InteractiveDynamics
using CairoMakie

mutable struct KoalaOrEucalyptus <: AbstractAgent
    id::Int
    type::Symbol #:koala or :Eucalyptus
    death_prob::Float64 #only for Koala
    production_rate::Float64 #only for eucalyptus
    consume_rate::Float64 #only for eucalyptus
    #reproduction prob for later
end

Koala(id,death_prob,production_rate, consume_rate) = KoalaOrEucalyptus(id, :koala, death_prob, production_rate, consume_rate)
Eucalyptus(id,death_prob,production_rate, consume_rate) = KoalaOrEucalyptus(id, :eucalyptus, death_prob, production_rate, consume_rate)



function initialize_model(;
    n_koala = 5,
    n_eucalyptus = 200,
    koala_death_rate = 8e-3,
    eucalyptus_production_rate = 0.009,
    eucalyptus_consume_rate = 0.05,
    seed = 23182,
     )
     properties = Dict(:euca_pr => eucalyptus_production_rate, :koala_dr => koala_death_rate)
     #properties = ()

     rng = MersenneTwister(seed)
     model = ABM(KoalaOrEucalyptus; properties,rng, scheduler = Schedulers.randomly)

     id = 0;

     for _ in 1:n_koala
         id += 1
         koala = Koala(id, koala_death_rate, 0,0)
         add_agent!(koala, model)
     end

     for _ in 1:n_eucalyptus
         id += 1
         eucalyptus = Eucalyptus(id, 0, eucalyptus_production_rate, eucalyptus_consume_rate)
         add_agent!(eucalyptus, model)
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
     #perhaps change the order of this operation
     #so that they can die before reproduce
     #add return to kill_agent
     if rand() < koala.death_prob
         kill_agent!(koala,model)
     end

 end

function eucalyptus_step!(eucalyptus,model)
    num_koalas = koalas(model) #num koalas
    num_eucalyptus = leaf(model) #num of eucalyptus
    prob_contact = num_koalas/(num_eucalyptus + num_koalas) #determine prob of contact
    if rand() < prob_contact
        if rand() < eucalyptus.consume_rate
            kill_agent!(eucalyptus,model) #the eucalyptus gets eaten
            id = nextid(model)
            koala = Koala(id, model.koala_dr, 0,0) #new koala
            add_agent!(koala, model) #new koala gets added to model
        end
    end
    new_euca = leaf(model) #recalculate num of eucalyptus
    if rand() < eucalyptus.production_rate*(1-(new_euca)/1000)
                #new eucalyptus if the above condition is met
                eucalyptus = Eucalyptus(nextid(model), 0, eucalyptus.production_rate, eucalyptus.consume_rate)
                add_agent!(eucalyptus, model) #new agent in model
            end
end


 #
 # function koala_eat!(koala, model)
 #     food = random_agent(model); #has eucalyptus
 #     if food.type == :eucalyptus
 #         if rand() <= food.consume_rate #change this to function of how much #times by prop of eucalyptus
 #             kill_agent!(food,model)
 #             reproduce!(koala, model)
 #         end
 #     end
 # end
 #
 # function reproduce!(agent, model)
 #     id = nextid(model)
 #     offspring = KoalaOrEucalyptus(
 #         id,
 #         agent.type,
 #         agent.death_prob,
 #         agent.production_rate,
 #         agent.consume_rate,
 #     )
 #     add_agent!(offspring,model)
 #     return
 # end

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
