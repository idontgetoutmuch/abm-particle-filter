using Agents, Random
using InteractiveDynamics
using CairoMakie

mutable struct Koalas <: AbstractAgent
    id::Int
    pos::Dims{2}
    type::Symbol
    #energy::Float64
    reproduction_prob::Float64
    death_prob::Float64
    #Δenergy::Float64
end


Koala(id, pos, repr, dpr) = Koalas(id, pos, :koala, repr, dpr)


function initialize_model(;
    n_koala = 50,
    dims = (20, 20),
    regrowth_time = 20,
    #Δenergy_koala = 50,
    koala_death = 0.01,
    koala_reproduce = 0.1,
    #human_reproduce = 0.05,
    seed = 23182,
)

    rng = MersenneTwister(seed)
    space = GridSpace(dims, periodic = false)
    properties = (
        fully_grown = falses(dims),
        countdown = zeros(Int, dims),
        regrowth_time = regrowth_time,
    )
    model = ABM(Koalas, space; properties, rng, scheduler = Schedulers.randomly)
    id = 0
    for _ in 1:n_koala
        id += 1
        #energy = rand(1:(Δenergy_koala*2)) - 1
        koala = Koala(id, (0, 0),koala_reproduce, koala_death)
        add_agent!(koala, model)
    end

    for p in positions(model) # random eucalyptus initial growth
        fully_grown = rand(model.rng, Bool)
        countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
        model.countdown[p...] = countdown
        model.fully_grown[p...] = fully_grown
    end
    return model
end


function model_step!(agent::Koalas, model)
    koala_step!(agent,model);
    # if agent.type == :human
    #     human_step!(agent, model)
    # else # then `agent.type == :wolf`
    #     koala_step!(agent, model)
    # end
end

function koala_step!(koala, model)
    walk!(koala, rand, model)
    # koala.energy -= 1
    koala_eat!(koala, model)
    if rand(model.rng) <= koala.death_prob
        kill_agent!(koala, model)
        return
    end
end




function koala_eat!(koala, model)
    if model.fully_grown[koala.pos...]
        ##change
        reproduce!(koala,model)
        #to reproduce based on probability
        # if rand(model.rng) <= koala.reproduction_prob
        #     reproduce!(koala, model)
        # end
        #koala.energy += koala.Δenergy
        model.fully_grown[koala.pos...] = false
    end
end



function reproduce!(agent, model)
    #agent.energy /= 2
    id = nextid(model)
    offspring = Koalas(
        id,
        agent.pos,
        agent.type,
        agent.reproduction_prob,
        agent.death_prob,
    )
    add_agent_pos!(offspring, model)
    return
end


function eucalyptus_step!(model)
    @inbounds for p in positions(model) # we don't have to enable bound checking
        if !(model.fully_grown[p...])
            if model.countdown[p...] ≤ 0
                model.fully_grown[p...] = true
                model.countdown[p...] = model.regrowth_time
            else
                model.countdown[p...] -= 1
            end
        end
    end
end

model = initialize_model()

koala(a) = a.type == :koala
count_eucalyptus(model) = count(model.fully_grown)


model = initialize_model()
n_steps = 500
adata = [(koala, count)]
mdata = [count_eucalyptus]
adf, mdf = run!(model, model_step!, eucalyptus_step!, n_steps; adata, mdata)

function plot_population_timeseries(adf, mdf)
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
    koalal = lines!(ax, adf.step, adf.count_koala, color = :blue)
    #humanl = lines!(ax, adf.step, adf.count_human, color = :orange)
    eucalyptusl = lines!(ax, mdf.step, mdf.count_eucalyptus, color = :green)
    figure[1, 2] = Legend(figure, [koalal, eucalyptusl], ["Koalas", "Eucalyptus"])
    figure
end

plot_population_timeseries(adf, mdf)
