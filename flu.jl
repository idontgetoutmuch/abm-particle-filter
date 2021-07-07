using Agents, Random, DataFrames, LightGraphs
using Distributions: Poisson, DiscreteNonParametric
using DrWatson: @dict
using Plots
using Random

#how many students
#num_students = 763

#Agent type
mutable struct Student <: AbstractAgent
    id::Int
    status::Symbol  # 1: S, 2: I, 3:R
end

function model_initialize(; num_students = 763, seed = 125)
    rng = Random.MersenneTwister(seed)
    #student agents with a random number generator
    model = ABM(Student)

     #add agents
    for i in 1:num_students
        add_agent!(model, :S)
    end

    ##infect a student
    agent = model[1]
    agent.status = :I

    return model
end

#create the model
model = model_initialize()


##the agent step function
#function agent_step!(agent,model)
    #transmit_and_recover!(agent,model)
#end


function model_step!(model)
    transmit_and_recover!(model)
end


##transmit function
function transmit_and_recover!(model)
    ##initally = 1
    total_infected(m) = count(a.status == :I for a in allagents(m))
    total_inf = total_infected(model)

    total_recovered(m) = count(a.status == :R for a in allagents(m))
    total_rec = total_recovered(model)

    total_sus(m) = count(a.status == :S for a in allagents(m))
    total_suspec = total_sus(model)

    beta = 2
    index = 0

    ##returns int value of number infected
    new_infected = rand(Poisson(total_suspec * total_inf * beta / 763))


    for a in allagents(model)
        if (index < new_infected)
            if a.status == :S
                a.status = :I
                index += 1
            end
        end
    end

Y = 0.5
counter = 0

##recover portion
    new_recover = rand(Poisson(total_inf*Y))

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


agent_df, model_df = run!(model,dummystep,model_step!,2)
