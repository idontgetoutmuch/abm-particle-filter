#dependencies of the model
using Agents, Random
using InteractiveDynamics
using CairoMakie
using Plots
using Statistics
using Distributions

mutable struct KoalaOrEucalyptus <: AbstractAgent
    id::Int
    pos::NTuple{2,Int}
    type::Symbol #:koala or :Eucalyptus
    death_prob::Float64 #only for Koala
    production_rate::Float64 #only for eucalyptus
    consume_rate::Float64 #only for eucalyptus
end

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
leaf(m) = count(a.type == :eucalyptus for a in allagents(m)) #for eucalyptus agents
koalas(m) = count(a.type == :koala for a in allagents(m)) #for koala agents


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

    #to ensure only one event occurs, use the death_prob
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

 koala_death_rate = 0.0025 #death rate of koala agents
eucalyptus_production_rate = 0.008 #production rate of eucalyptus
eucalyptus_consume_rate = 0.02 #death/consumption rate of eucalyptus

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
    figure[1, 2] = Legend(figure, [koalal, eucalyptusl], ["Koalas", "E"])
    figure
end

plot_population_timeseries(adf)


# start of particle filter functions 
t_koala_death_rate = 0.0025
t_eucalyptus_production_rate = 0.008
t_eucalyptus_consume_rate = 0.02

# Parameters to be estimated
μ = [t_koala_death_rate, t_eucalyptus_production_rate, t_eucalyptus_consume_rate];


Qkdr  = 0.0002 #covariance for koala_death_rate
Qepr    = 0.0005 #covariance for eucalyptus_production_rate
Qecr = 0.0002 #covariance for eucalyptus_consume_rate

# Parameter update covariance aka parameter diffusivity
var = [[Qkdr, 0.0, 0.0] [0.0,  Qepr , 0.0] [0.0,  0.0, Qecr]];

# Observation covariance
R = Matrix{Float64}(undef,1,1); 
R[1,1] = 0.1;

# Number of particles
P = 50;

#For data to be tracked
N = 10; #number of koalas
actuals = adf[:,2]; #let actuals be the current run results

#initalise the various models
ABM_params = [koala_death_rate;eucalyptus_production_rate;eucalyptus_consume_rate]
templates = [initialize_model(input_params = ABM_params) for n in 1:P]

function resample_stratified(weights)

    N = length(weights)
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


 function measure(x)
    num_k = koalas(x);
    return num_k;
end


function pf_init(inits, g, N, y, R)

    y_pf = zeros(Int64,N);
    log_w = zeros(N);

    y_pf = map(g, inits)
    log_w = map(x -> logpdf(MvNormal([y], R), x), map(x -> [x], y_pf))

    return(y_pf, log_w)

end

function pf(inits, g, log_w, N, y, R)

    wn = zeros(N);
    jnits = [initialize_model(input_params = ABM_params) for n in 1:P];
    y_pf = zeros(Int64,N);

    wn = exp.(log_w .- maximum(log_w));
    swn = sum(wn);
    wn .= wn ./ swn;

    a = resample_stratified(wn);

    for i in 1:N
        jnits[i] = deepcopy(inits[a[i]])
    end

    for i in 1:N
        Agents.step!(jnits[i], agent_step!, 1)
        y_pf[i] = g(jnits[i])
    end

    log_w = map(x -> logpdf(MvNormal([y], R), x), map(x -> [x], y_pf))

    max_weight = maximum(log_w);
    wn = exp.(log_w .- max_weight);
    swn = sum(wn);
    wn .= wn ./ swn;
    predictive_likelihood = max_weight + log(swn) - log(N);

    return(y_pf, log_w, predictive_likelihood, jnits)

end

function runPf(inits, g, init_log_weights, actuals, predicted1, R)
    l         = length(actuals)
    predicted = zeros(l);
    log_w     = zeros(l, P);

    predicted[1] = predicted1;
    log_likelihood = 0;

    for i in 2:l
        (obs, new_log_weights, predictive_likelihood, news) = pf(deepcopy(inits), g, init_log_weights, P, map(x -> convert(Float64,x), actuals[i]), R);
        predicted[i] = mean(obs);
        log_likelihood = log_likelihood + predictive_likelihood;

        inits            = news;
        init_log_weights = new_log_weights;
    end
    return predicted, log_likelihood;
end


function particleFilter(templates, g, P, actuals, R)
    l = length(actuals);

    inits = deepcopy(templates);
    (initial_end_states, init_log_weights) = pf_init(inits, g, P, map(x -> convert(Float64,x), actuals[1]), R);

    predicted, log_likelihood = runPf(inits, g, init_log_weights, actuals, mean(initial_end_states), R);

    return predicted, log_likelihood;
end


@time predicted = particleFilter(templates, measure, P, actuals, R);

l = length(actuals)

Plots.plot(1:l, actuals, label="Actual", color = "red", lw = 3, title = string("Results from ", P, " particles\n"), xlab="Time",ylabel="Number", legend = :left)
Plots.plot!(1:l, predicted, label="Tracked by Prior Particle Filter", color = "blue", lw = 3)


function prior_sample(μ, var)
    rand(MvLogNormal(log.(μ), var))
end

function log_prior_pdf(x, μ, var)
    logpdf(MvLogNormal(log.(μ), var), x)
end

function pmh(g, P, N, K, μ, var, actuals, R,num_params, theta)
    prop_acc            = zeros(K);
    log_likelihood_curr = -Inf;


    log_prior_curr = log_prior_pdf(theta[:, 1], μ, var);

    for k = 2:K
        theta_prop  = prior_sample(theta[:, k - 1], var);
        theta[:, k] = theta_prop;

        inits = [initialize_model(input_params = theta[:,k]) for n in 1:P];
        predicted, log_likelihood_prop = particleFilter(inits, g, P, actuals, R);
        log_likelihood_diff = log_likelihood_prop - log_likelihood_curr;

        log_prior_curr = log_prior_pdf(theta[:, k - 1], μ, var);
        log_prior_prop = log_prior_pdf(theta[:, k],     μ, var);
        log_prior_diff = log_prior_prop - log_prior_curr;

        acceptance_prob = exp(log_prior_diff + log_likelihood_diff);

        r = rand();
        if (r < acceptance_prob)
            log_likelihood_curr = log_likelihood_prop;
            prop_acc[k]         = 1;
        else
            theta[:, k] = theta[:, k - 1];
            prop_acc[k] = 0;
        end

        print("#####################################################################\n");
        print(" Iteration: ", k, " completed.\n");
        print(" Current state of the Markov chain: ", theta[:, k], "\n");
        print(" Proposed state of the Markov chain: ", theta_prop, "\n");
        print(" Current posterior mean: ", mean(theta[:, 1:k], dims = 2), "\n");
        print(" Current acceptance: ", mean(prop_acc[1:k]), "\n");
        print("#####################################################################\n");


    end
end

K = 100 #Number of posterior particle filter runs 
num_params = 3; #number of parameters to be estimated
theta = zeros(num_params, K); #initalised theta or parameter vector
theta[:, 1] = prior_sample(μ, var); #detered parameters 

pmh(measure, P, N, K, μ, var, actuals, R, num_params, theta)


#This mean of the posterior was used
mean_post = [0.0024999874252689897; 0.008035042405172334; 0.019789952330170887]
# For refrence, the original parameter set was [0.0025, 0.008, 0.02]


predicted_model = [initialize_model(input_params = mean_post) for n in 1:P]

@time predicted_posterior = particleFilter(predicted_model, measure, P, actuals, R);

Plots.plot(1:l, actuals, label="Actual", color = "red", lw = 3, title = string("Koala Results from ", P, " particles\n"), xlab="Time",ylabel="Number", legend = :topright,legendfontsize = 9)
Plots.plot!(1:l, predicted, label="Tracked by Prior Particle Filter", color = "blue", lw = 3)
Plots.plot!(1:l, predicted_posterior, label="Tracked by Posterior Particle Filter", color = "green", lw = 3)

