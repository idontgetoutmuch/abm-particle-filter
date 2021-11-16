using Agents, Random, DataFrames, LightGraphs
using Distributions
using DrWatson: @dict
using Plots
using Random
using InteractiveDynamics
using CairoMakie
using Statistics: mean
using DifferentialEquations
using SimpleDiffEq
using StatsPlots
using BlackBoxOptim, Random

using Distributions


function sir_ode!(du,u,p,t)
    (S, I, R) = u
    (β, γ)    = p
    N = S + I + R
    @inbounds begin
        du[1] = -β * I / N * S
        du[2] =  β * I / N * S - γ * I
        du[3] =                  γ * I
    end
    nothing
end;


δt = 0.1
tmax = 15.0
tspan = (0.0, tmax)
t = 1.0 : δt : tmax;

u0 = [762.0, 1.0, 0.0]; # S, I, R
p = [2.0, 0.5]; # β, γ
actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]


prob_ode = ODEProblem(sir_ode!, u0, tspan, p);
sol_ode = solve(prob_ode);

df_ode = DataFrame(Tables.table(sol_ode(t)'))
df_ode[!,:t] = t;

p1 = [1.7, 0.5]; # β, γ

prob_ode1 = ODEProblem(sir_ode!, u0, tspan, p1);
sol_ode1 = solve(prob_ode1);

df_ode1 = DataFrame(Tables.table(sol_ode1(t)'))
df_ode1[!,:t] = t;

Plots.plot(df_ode1[:,:t],df_ode1[:,1],label="S",xlab="Time",ylabel="Number", title = "ODE solution")
Plots.plot!(df_ode1[:,:t],df_ode1[:,2],label="I")
Plots.plot!(df_ode1[:,:t],df_ode1[:,3],label="R")
Plots.plot!(1:15, actuals, label="A")

## agent type
mutable struct Student <: AbstractAgent
    id::Int
    status::Symbol  # 1: S, 2: I, 3:R
end

## function to calculate gamma
function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end;



##model function
function init_model(β :: Float64, c :: Float64, γ :: Float64, N :: Int64, I0 :: Int64)
    properties = @dict(β,c,γ)
    model = ABM(Student; properties=properties)
    for i in 1 : N #for all students
        if i <= I0 # infect initial number of students
            s = :I
        else
            s = :S # rest of students are susceptible
        end
        p = Student(i,s) #create student agent type
        p = add_agent!(p,model) #add this student to ABM
    end
    return model
end;

##Agent step function

function agent_step!(agent, model)
    transmit!(agent, model)
    recover!(agent, model)
end;

function transmit!(agent, model)
    # If I'm not susceptible, I return
    agent.status != :S && return
    #based on c value, decide contacts
    ncontacts = rand(Poisson(model.properties[:c]))
    for i in 1:ncontacts
        # Choose random individual
        alter = random_agent(model)
        # if random agent is infected, and rand is <=
        # beta (infection rate)
        if alter.status == :I && (rand() ≤ model.properties[:β])
            # An infection occurs
            agent.status = :I
            break
        end
    end
end;


function recover!(agent, model)
    #if agent is not infected, then returns
    agent.status != :I && return
    #if random probability of recovery <= gamma
    if rand() ≤ model.properties[:γ]
        #then recover agent
            agent.status = :R
    end
end;

##functions to determine S,I,R
susceptible(x) = count(i == :S for i in x)
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x);

##model parameters
δt = 0.1
nsteps = 150
tf = nsteps * δt #t final
t = 0 : δt : tf; #time vector

# β = 0.05
β = 0.25
# c = 10.0 * δt
c = 7.5 * δt #contact rate
# γ = rate_to_proportion(0.25, δt);
γ = rate_to_proportion(0.50, δt); #determine gamma

# N = 1000
N = 763 #total number of students
# I0 = 10;
I0 = 1; #one person infected

actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]

Random.seed!(1234);

#create model




## black box testing
total_infected(m) = count(a.status == :I for a in allagents(m))

# black box optimise
function myCost(x, ys)
    model = init_model(
        x[1],
        x[2],
        x[3],
        N,
        I0,

    )

    _, data = run!(model,agent_step!,20;
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
    2.0,
    0.75,
    0.5,
    N,
    I0,
]

#actual values
actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]



#part cost of actuals vs simulation values
partCost(actuals)(x0)


# optimise result
result = bboptimize(partCost(actuals),
    SearchRange = [
        (1.0, 5.0), #2
        (0.75,1), #0.75
        (0.4, 1.0), #0.5

   ],
   NumDimensions = 3,
   MaxTime = 50,
                    )

 best_fitness(result)

 x = best_candidate(result)

 Random.seed!(1234)

#model generated from optimised values
 # model_op = init_model(;
 #     β = x[1],
 #     c = x[2],
 #     γ = x[3],
 # )
abm_model = init_model(β, c, γ, N, I0)

model_op = init_model(x[1], x[2], x[3], N, I0);
op_data, _ = run!(model_op, agent_step!, nsteps; adata = to_collect);
og_data, _ = run!(abm_model, agent_step!, nsteps; adata = to_collect);

op_data[!,:t] = t;
og_data[!,:t] = t;

#Plots.plot(t,abm_data[:,2],label="S",xlab="Time",ylabel="Number")
Plots.plot(t,op_data[:,3],label="I from black box",xlab="Time",ylabel="Number")
#Plots.plot!(t, og_data[:,3], label = "I normal run")
Plots.plot!(1:15, actuals, label="A")


# Plots.plot(model_df_2.step, model_df_2.total_infected, labels = "Optimised Version", legend = :left)
# #actual version
# Plots.plot!(model_df_2.step, actuals[1:11], title = "Total population infected", labels = "Actual version", legend = :left, xlabel = "Day", ylabel = "Number of cases")
