#new Koalas
using Agents, Random
using InteractiveDynamics
using CairoMakie
using Plots
using Statistics
using Distributions

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
    n_koala = 10,
    n_eucalyptus = 20,
    input_params:: Vector{Float64},
    seed = 23182,
     )
     koala_death_rate = input_params[1] #death rate of koala agents
     eucalyptus_production_rate = input_params[2]#production rate of eucalyptus
     eucalyptus_consume_rate = input_params[3] #death/consumption rate of eucalyptus
     #properties = @dict(eucalyptus_production_rate)
     properties = Dict(:euca_pr => eucalyptus_production_rate, :koala_dr => koala_death_rate, :euca_cr => eucalyptus_consume_rate)

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


#helper functions to count amount koala and eucalyptus agents
leaf(m) = count(a.type == :eucalyptus for a in allagents(m))
koalas(m) = count(a.type == :koala for a in allagents(m))


 function agent_step!(agent::KoalaOrEucalyptus,model)
    # print("#####################################################################\n");
    # print(" Iteration: ")
     # num_euc = leaf(model)
     # if num_euc < 1
     #     eucalyptus_step!(agent,model)
     if agent.type == :koala
         koala_step!(agent,model)
    else
        eucalyptus_step!(agent,model)
    end

    num_euca = leaf(model)
    num_koalas = koalas(model)

    if num_euca == 0
        if rand() < model.euca_pr
            id = nextid(model)
            eucalyptus_new = Eucalyptus(id, 0,  model.euca_pr, model.euca_cr)
            add_agent!(eucalyptus_new, model)
        end
    end

        #if there are no koala agents in simulation, then
    #based on the eucalyptus consumption rate, add a koala agent to sim
    if num_koalas == 0
        if rand() < model.euca_cr
            id = nextid(model)
            koala_new = Koala(id, model.koala_dr, 0,0)
            add_agent!(koala_new, model)
        end
    end


 end

 function koala_step!(koala,model)
     #perhaps change the order of this operation
     #so that they can die before reproduce
     #add return to kill_agent
     if rand() < koala.death_prob
         kill_agent!(koala,model)
     else
        koala_eat!(koala,model)
     end

 end

function eucalyptus_step!(eucalyptus,model)
    new_euca = leaf(model) #calculate num of eucalyptus
    if rand() < eucalyptus.production_rate*(1-(new_euca)/1000)
                #new eucalyptus if the above condition is met
                id = nextid(model)
                eucalyptus_new = Eucalyptus(id, 0, model.euca_pr, model.euca_cr)
                add_agent!(eucalyptus_new, model)
    end
end


 function koala_eat!(koala, model)
     food = random_agent(model); #has eucalyptus
     if food.type == :eucalyptus
         if rand() < food.consume_rate #change this to function of how much #times by prop of eucalyptus
             kill_agent!(food,model)
             id = nextid(model)
             koala_new = Koala(id, model.koala_dr, 0,0)
             add_agent!(koala_new, model)
         end
     end
 end

 
 koala(a) = a.type == :koala
 eucalyptus(a) = a.type == :eucalyptus


 koala_death_rate = 0.0025 #death rate of koala agents
eucalyptus_production_rate = 0.008 #production rate of eucalyptus
eucalyptus_consume_rate = 0.02 #death/consumption rate of eucalyptus

ABM_params = [koala_death_rate;eucalyptus_production_rate;eucalyptus_consume_rate]

 model = initialize_model(input_params = ABM_params)
 n_steps = 2000
 adata = [(koala, count), (eucalyptus, count)]
 adf, _ = run!(model, agent_step!, dummystep, n_steps; adata)

 function plot_population_timeseries(adf)
     figure = Figure(resolution = (600, 400))
     ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
     koalal = lines!(ax, adf.step, adf.count_koala, color = :blue)
     eucalyptusl = lines!(ax, adf.step, adf.count_eucalyptus, color = :green)
     figure[1, 2] = Legend(figure, [koalal, eucalyptusl], ["Koalas", "E"])
     figure
 end

 plot_population_timeseries(adf)


#  ## plots

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
 Plots.plot(adf.step, a_koalas[:,1],  label = "", title = "ABM populations from 100 runs (new model)",xlab="Time step",ylabel="Population", linecolor = :red, titlefontsize = 10)
 Plots.plot!(adf.step, a_euca[:,1], label = "",xlab="Time step",ylabel="Population", linecolor = :blue)
 for i in 2:(size(a_koalas)[2]-1)
     #Plots.plot!(t, a_50[:,i])#, seriestype = :scatter)
     Plots.plot!(adf.step, a_koalas[:,i], linecolor = :red, label = "")
     Plots.plot!(adf.step, a_euca[:,i], linecolor = :blue, label = "")

 end
 #Plots.plot!(t,a_50[:,50])#, seriestype = :scatter)
 Plots.plot!(adf.step, a_koalas[:,size(a_koalas)[2]], label = "Koala", linecolor = :red)
 Plots.plot!(adf.step, a_euca[:,size(a_koalas)[2]],label = "Eucalyptus", linecolor = :blue, legend = :topleft, legendfontsize = 6)


 ### particle filter

 # start of pf functions

#  function prior_sample(μ, var)
#     rand(MvLogNormal(log.(μ), var))
# end

# function log_prior_pdf(x, μ, var)
#     logpdf(MvLogNormal(log.(μ), var), x)
# end

# function measure(x)
#     num_k = koalas(x);
#     return num_k;
#     #return modelCounts(x)[3]
# end


# function resample_stratified(weights)

#    N = length(weights)
#    positions =  (rand(N) + collect(range(0, N - 1, length = N))) / N

#    indexes = zeros(Int64, N)
#    cumulative_sum = cumsum(weights)
#    i, j = 1, 1
#    while i <= N
#        if positions[i] < cumulative_sum[j]
#            indexes[i] = j
#            i += 1
#        else
#            j += 1
#        end
#    end
#    return indexes
# end

# function pf_init(inits, g, N, y, R)

#     y_pf = zeros(Int64,N);
#     log_w = zeros(N);

#     y_pf = map(g, inits)
#     log_w = map(x -> logpdf(MvNormal([y], R), x), map(x -> [x], y_pf))

#     return(y_pf, log_w)

# end

# #furture - move junts to outside
# function pf(inits, g, log_w, N, y, R)

#     wn = zeros(N);
#     jnits = [initialize_model(input_params = ABM_params) for n in 1:P]; #[init_model(β, c, γ, N, 1) for n in 1:P]

#     y_pf = zeros(Int64,N);

#     wn = exp.(log_w .- maximum(log_w));
#     swn = sum(wn);
#     wn .= wn ./ swn;

#     a = resample_stratified(wn);

#     for i in 1:N
#         jnits[i] = deepcopy(inits[a[i]])
#     end

#     # Can this be made parallel?
#     for i in 1:N
#         Agents.step!(jnits[i], agent_step!, 1)
#         y_pf[i] = g(jnits[i])
#     end

#     log_w = map(x -> logpdf(MvNormal([y], R), x), map(x -> [x], y_pf))

#     max_weight = maximum(log_w);
#     wn = exp.(log_w .- max_weight);
#     swn = sum(wn);
#     wn .= wn ./ swn;
#     predictive_likelihood = max_weight + log(swn) - log(N);

#     return(y_pf, log_w, predictive_likelihood, jnits)

# end

# function runPf(inits, g, init_log_weights, actuals, predicted1, R)
#     l         = length(actuals)
#     predicted = zeros(l);
#     log_w     = zeros(l, P);

#     predicted[1] = predicted1;
#     log_likelihood = 0;

#     for i in 2:l
#         (obs, new_log_weights, predictive_likelihood, news) = pf(deepcopy(inits), g, init_log_weights, P, map(x -> convert(Float64,x), actuals[i]), R);
#         predicted[i] = mean(obs);
#         log_likelihood = log_likelihood + predictive_likelihood;

#         inits            = news;
#         init_log_weights = new_log_weights;
#     end
#     return predicted, log_likelihood;
# end

# function particleFilter(templates, g, P, actuals, R)
#     l = length(actuals);

#     inits = deepcopy(templates);
#     (initial_end_states, init_log_weights) = pf_init(inits, g, P, map(x -> convert(Float64,x), actuals[1]), R);

#     predicted, log_likelihood = runPf(inits, g, init_log_weights, actuals, mean(initial_end_states), R);

#     return predicted, log_likelihood;
#    end

# #run pf 

#  koala(a) = a.type == :koala
# eucalyptus(a) = a.type == :eucalyptus


# koala_death_rate = 0.0025 #death rate of koala agents
# eucalyptus_production_rate = 0.008 #production rate of eucalyptus
# eucalyptus_consume_rate = 0.02 #death/consumption rate of eucalyptus

# ABM_params = [koala_death_rate;eucalyptus_production_rate;eucalyptus_consume_rate]


# #for recreation data
# model = initialize_model(input_params = ABM_params)
# n_steps = 100
# adata = [(koala, count), (eucalyptus, count)]
# adf, _ = run!(model, agent_step!, dummystep, n_steps; adata)

#  N = 10; #koalas
#  actuals = adf[:,2]; #let actuals be the current run

#  R = Matrix{Float64}(undef,1,1);
#  R[1,1] = 0.1;

#  P = 100; #how many models
#  templates = [initialize_model(input_params = ABM_params) for n in 1:P]

#  @time predicted = particleFilter(templates, measure, P, actuals, R);

# l = length(actuals)

# Plots.plot(1:l, actuals, label="Actual", color = "red", lw = 3, title = string("Results from ", P, " particles\n"), xlab="Time",ylabel="Number", legend = :left)
# Plots.plot!(1:l, predicted, label="Tracked by Prior Particle Filter", color = "blue", lw = 3)


# t_koala_death_rate = 0.0025
# t_eucalyptus_production_rate = 0.008
# t_eucalyptus_consume_rate = 0.02

# μ = [t_koala_death_rate, t_eucalyptus_production_rate, t_eucalyptus_consume_rate]

#  var = [[0.0002, 0.0, 0.0] [0.0,  0.0005, 0.0] [0.0,  0.0, 0.0002]];



#  function pmh(g, P, N, K, μ, var, actuals, R,num_params, theta)
#      # This need generalising - in this case we have 3 parameters but
#      # we should handle any number
#      prop_acc            = zeros(K);
#      log_likelihood_curr = -Inf;


#      log_prior_curr = log_prior_pdf(theta[:, 1], μ, var);

#      for k = 2:K
#          theta_prop  = prior_sample(theta[:, k - 1], var);
#          theta[:, k] = theta_prop;
#          # β = theta[1, k];
#          # c = theta[2, k];
#          # γ = theta[3, k];

#          inits = [initialize_model(input_params = theta[:,k]) for n in 1:P];
#          predicted, log_likelihood_prop = particleFilter(inits, g, P, actuals, R);
#          log_likelihood_diff = log_likelihood_prop - log_likelihood_curr;

#          log_prior_curr = log_prior_pdf(theta[:, k - 1], μ, var);
#          log_prior_prop = log_prior_pdf(theta[:, k],     μ, var);
#          log_prior_diff = log_prior_prop - log_prior_curr;

#          acceptance_prob = exp(log_prior_diff + log_likelihood_diff);

#          r = rand();
#          if (r < acceptance_prob)
#              log_likelihood_curr = log_likelihood_prop;
#              prop_acc[k]         = 1;
#          else
#              theta[:, k] = theta[:, k - 1];
#              prop_acc[k] = 0;
#          end

#          print("#####################################################################\n");
#          print(" Iteration: ", k, " completed.\n");
#          print(" Current state of the Markov chain: ", theta[:, k], "\n");
#          print(" Proposed state of the Markov chain: ", theta_prop, "\n");
#          print(" Current posterior mean: ", mean(theta[:, 1:k], dims = 2), "\n");
#          print(" Current acceptance: ", mean(prop_acc[1:k]), "\n");
#          print("#####################################################################\n");


#      end
#  end


# #second part of pf
#  K = 100;
#  num_params = 3;
#  theta = zeros(num_params, K);
#  theta[:, 1] = prior_sample(μ, var);



# pmh(measure, P, N, K, μ, var, actuals, R, num_params, theta)

# Random.seed!(1234);

# #  #[0.004845152098288908; 0.007914075100948296; 0.041591724585527394]

# #This mean of the posterior was used
# #mean_post =  [0.0025185423968039694; 0.007990037644470045; 0.019872499092052735]
# mean_post = [0.0024999874252689897; 0.008035042405172334; 0.019789952330170887]


# predicted_model = [initialize_model(input_params = mean_post) for n in 1:P]

# @time predicted_posterior = particleFilter(predicted_model, measure, P, actuals, R);

# Plots.plot(1:l, actuals, label="Actual", color = "red", lw = 3, title = string("Koala Results from ", P, " particles\n"), xlab="Time",ylabel="Number", legend = :topleft,legendfontsize = 9)
# Plots.plot!(1:l, predicted, label="Tracked by Prior Particle Filter", color = "blue", lw = 3)
# Plots.plot!(1:l, predicted_posterior, label="Tracked by Posterior Particle Filter", color = "green", lw = 3)