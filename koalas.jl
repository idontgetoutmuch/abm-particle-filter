using Agents, Random
using InteractiveDynamics
using CairoMakie

mutable struct HumansOrKoalas <: AbstractAgent
    id::Int
    pos::Dims{2}
    type::Symbol # :sheep or :wolf
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
end


Human(id, pos, energy, repr, Δe) = HumansOrKoalas(id, pos, :human, energy, repr, Δe)
Koala(id, pos, energy, repr, Δe) = HumansOrKoalas(id, pos, :koala, energy, repr, Δe)



function initialize_model(;
    n_koala = 50,
    n_human = 100,
    dims = (20, 20),
    regrowth_time = 20,
    Δenergy_koala = 50,
    Δenergy_human = 10,
    koala_reproduce = 0.1,
    human_reproduce = 0.05,
    seed = 23182,
)

    rng = MersenneTwister(seed)
    space = GridSpace(dims, periodic = false)
    properties = (
        fully_grown = falses(dims),
        countdown = zeros(Int, dims),
        regrowth_time = regrowth_time,
    )
    model = ABM(HumansOrKoalas, space; properties, rng, scheduler = Schedulers.randomly)
    id = 0
    for _ in 1:n_koala
        id += 1
        energy = rand(1:(Δenergy_koala*2)) - 1
        koala = Koala(id, (0, 0), energy, koala_reproduce, Δenergy_koala)
        add_agent!(koala, model)
    end
    for _ in 1:n_human
        id += 1
        energy = rand(1:(Δenergy_human*2)) - 1
        human = Human(id, (0, 0), energy, human_reproduce, Δenergy_human)
        add_agent!(human, model)
    end
    for p in positions(model) # random eucalyptus initial growth
        fully_grown = rand(model.rng, Bool)
        countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
        model.countdown[p...] = countdown
        model.fully_grown[p...] = fully_grown
    end
    return model
end


function humanorkoala_step!(agent::HumansOrKoalas, model)
    if agent.type == :human
        human_step!(agent, model)
    else # then `agent.type == :wolf`
        koala_step!(agent, model)
    end
end

function koala_step!(koala, model)
    walk!(koala, rand, model)
    koala.energy -= 1
    koala_eat!(koala, model)
    if koala.energy < 0
        kill_agent!(koala, model)
        return
    end
    if rand(model.rng) <= koala.reproduction_prob
        reproduce!(koala, model)
    end
end

function human_step!(human, model)
    walk!(human, rand, model)
    human.energy -= 2 #more costly
    # fix here
    #agents = collect(agents_in_position(human.pos, model))
    #dinner = filter!(x -> x.type == :sheep, agents)
    human_eat!(human, model)
    if human.energy < 0
        kill_agent!(human, model)
        return
    end
    if rand(model.rng) <= human.reproduction_prob
        reproduce!(human, model)
    end
end


function koala_eat!(koala, model)
    if model.fully_grown[koala.pos...]
        koala.energy += koala.Δenergy
        model.fully_grown[koala.pos...] = false
    end
end

function human_eat!(human, model)
    if model.fully_grown[human.pos...]
        human.energy += human.Δenergy
        model.fully_grown[human.pos...] = false
    end
    # if !isempty(sheep)
    #     dinner = rand(model.rng, sheep)
    #     kill_agent!(dinner, model)
    #     wolf.energy += wolf.Δenergy
    # end
end


function reproduce!(agent, model)
    agent.energy /= 2
    id = nextid(model)
    offspring = HumansOrKoalas(
        id,
        agent.pos,
        agent.type,
        agent.energy,
        agent.reproduction_prob,
        agent.Δenergy,
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
human(a) = a.type == :human
count_eucalyptus(model) = count(model.fully_grown)


model = initialize_model()
n_steps = 500
adata = [(koala, count), (human, count)]
mdata = [count_eucalyptus]
adf, mdf = run!(model, humanorkoala_step!, eucalyptus_step!, n_steps; adata, mdata)

function plot_population_timeseries(adf, mdf)
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
    koalal = lines!(ax, adf.step, adf.count_koala, color = :blue)
    humanl = lines!(ax, adf.step, adf.count_human, color = :orange)
    eucalyptusl = lines!(ax, mdf.step, mdf.count_eucalyptus, color = :green)
    figure[1, 2] = Legend(figure, [koalal, humanl, eucalyptusl], ["Koalas", "Humans", "Eucalyptus"])
    figure
end

plot_population_timeseries(adf, mdf)
