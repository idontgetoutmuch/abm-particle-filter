using Pkg
pkg"activate ."

using Random
using Distributions
using Gadfly
using LinearAlgebra

# FIXME: This doesn't seem to give reproducibility
rng = MersenneTwister(1234);

T = 500;

deltaT = 0.01;
g  = 9.81;

qc1 = 0.0001;

bigQ = [ qc1 * deltaT^3 / 3 qc1 * deltaT^2 / 2;
         qc1 * deltaT^2 / 2       qc1 * deltaT
         ];

bigR  = [0.01];

x = zeros(T + 1, 2);
y = zeros(T,     1);

x[1, :] = [0.01 0];

for t = 2:T+1
    epsilon = rand(MvNormal(zeros(1), bigR));
    y[t - 1, :] = sin(x[t- 1, 1]) .+ epsilon
    x1 = x[t - 1, 1] + x[t - 1, 2] * deltaT;
    x2 = x[t - 1, 2] - g * sin(x[t - 1, 1]) * deltaT;
    eta = rand(MvNormal(zeros(2), bigQ));
    xNew = [x1, x2] .+ eta;
    x[t, :] = xNew;
end

plot(layer(y = x[1:T,1], Geom.line, Theme(default_color=color("red"))), layer(y = y[1:T,1], Geom.point))

function ff(x, g)
    x1 = x[1] + x[2] * deltaT;
    x2 = x[2] - g * sin(x[1]) * deltaT;
    [x1, x2];
end

function f_g(x, k)
    map(y -> ff(y, k), mapslices(y -> [y], x, dims = 1))
end

function h(x)
    map(z -> sin(z[1]), mapslices(y->[y], x, dims=1))
end

nx = 2;

n_th = 1;

# prior_pdf = @(theta) mvnpdf(theta,0,1)';
function prior_pdf(theta)
    pdf(MvNormal(zeros(length(theta)), I), theta)
end

# prior_sample = @(n) mvnrnd(zeros(n,1),1)';
function prior_sample(n)
    rand(MvNormal(zeros(n), I))
end

N = [5 30 80];
K = [100 1000 10000];
Km = [3 3 3];
N_th = [10 100 1000];

function resample_systematic( w )
    N = length(w);
    Q = cumsum(w);
    T = collect(range(0, stop = 1 - 1 / N, length = N)) .+ rand(1) / N;
    append!(T, 1);
    ix = zeros(Int64, (1, N));
    i=1;
    j=1;
    while (i <= N)
        if (T[i] < Q[j])
            ix[i] = j;
            i = i + 1;
        else
            j = j + 1;
        end
    end
    ix;
end

function resample_stratified( weights )

    N = length(weights)
    # make N subdivisions, and chose a random position within each one
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

function k(x)
    f_g(x, g)
end

# DJS to change this to accept HA's function which takes S I R beta
# gamma and returns S I R one time step ahead.
function pf(inits, N, f, h, y, Q, R, nx)
    # inits - initial values

    # N - number of particles

    # f is the state update function - for us this is the ABM - takes
    # a state and the parameters and returns a new state one time step
    # forward

    # h is the observation function - for us the state is S, I, R
    # (although S + I + R = total boys) but we can only observe I

    # y is the data - for us this the number infected for each of the
    # 14 days

    # To avoid particle collapse we need to perturb the parameters -
    # we do this by sampling from a multivariate normal distribution
    # with a given covariance matrix Q

    T = length(y)
    log_w = zeros(T,N);
    x_pf = zeros(nx,N,T);
    x_pf[:,:,1] = inits;
    wn = zeros(N);

    for t = 1:T
        if t >= 2
            a = resample_stratified(wn);

            # We need to change this line to update the state (S I R)
            # with the ABM and update the parameters by adding a
            # sample from a multivariate normal distribution - since
            # we have two parameters this is a 2 dimensional
            # multivariate normal distribution.

            # So take the S I R beta and gamma and produce a new S I R
            # using the ABM and produce new beta and gamma using a
            # random sample.

            # beta is about 2.0 and gamma is about 0.5 so we should
            # probably take the covariance matrix to be

            # [0.1 0.0; 0.0 0.01]

            x_pf[:, :, t] = hcat(f(x_pf[:, a, t-1])...) + rand(MvNormal(zeros(nx), Q), N)
        end

        # For now choose R to be [0.1]

        log_w[t, :] = logpdf(MvNormal(y[t, :], R), h(x_pf[:,:,t]));

        # To avoid underflow subtract the maximum before moving from
        # log space

        wn = map(x -> exp(x), log_w[t, :] .- maximum(log_w[t, :]));
        wn = wn / sum(wn);
    end

    log_W = sum(map(log, map(x -> x / N, sum(map(exp, log_w[1:T, :]), dims=2))));

    return(x_pf, log_w, log_W)

end

N = 50;

inits = zeros(nx, N);
inits[1, :] .= 0.01;
inits[2, :] .= 0.00;

(foo, bar, baz) = pf(inits, N, k, h, y, bigQ, bigR, nx);

plot(layer(y = x[1:T,1], Geom.line, Theme(default_color=color("red"))), layer(y = map(x -> x / 50, sum(foo[1,:,:],dims=1)), Geom.line))

function pmh( K, N, n_th, u, y, f_g, g, nx, prior_sample, prior_pdf, Q, R)

    theta = zeros(n_th,K+1);
    log_W = -Inf;

    while log_W==-Inf %Find an initial sample without numerical problems
        theta(:,1) = prior_sample(1);
        log_W = pf( N, @(x,u)f_g(x,u,theta(:,1)), g, u, y, Q, R, nx);
    end

    for k = 1:K
        theta_prop = theta(:,k) + 0.1*randn(n_th,1);
        log_W_prop = pf( N, @(x,u)f_g(x,u,theta_prop), g, u, y, Q, R, nx);
        dm = rand;
        mh_ratio = exp(log_W_prop-log_W)*prior_pdf(theta_prop)/prior_pdf(theta(:,k));
        if isnan(mh_ratio)
            alpha = 0;
        else
            alpha = min(1,mh_ratio);
        end
        if dm < alpha
            theta(:,k+1) = theta_prop;
            log_W = log_W_prop;
            new = true;
        else
            theta(:,k+1) = theta(:,k);
            new = false;
        end
        if new == true; display(['PMH Sampling ', num2str(k), ': Proposal accepted!']); else
            display(['PMH Sampling ', num2str(k), ': Proposal rejected']);
        end
    end
end


# for i = 1:3
#     tic
#     theta{i} = pmh( K(i), N(i), n_th, u, y, f_g, g, nx, prior_sample, prior_pdf, Q, R); %#ok
#     pmh_time(i) = toc; %#ok

#     tic
#     [ x_th1, w_th1 ] = smc2( Km(i), N_th(i), n_th, u, y, f_g, g, nx, prior_sample, prior_pdf, Q, R);
#     x_th{i} = x_th1; %#ok
#     w_th{i} = w_th1; %#ok
#     smc2_time(i) = toc; %#ok

# end

# %%

# hist_v = 0.05:0.05:0.85;

# figure(4), clf
# for i = 1:3
#     a = systematic_resampling(w_th{i}(:,T),N_th(i));
#     x_th_f = x_th{i}(:,a,T);
#     subplot(3,3,3*i-1)
#     [by,bx] = hist(x_th_f(1,:),hist_v);

#     ind = find(by>0,1,'first'):find(by>0,1,'last');
#     by = by(ind); bx = bx(ind);

#     th_ml = 0.54;

#     bdx = diff(bx); bdx = bdx(1);
#     hxy = [bx(1)-bdx/2 0];
#     by = by/N_th(i);

#     for j=1:numel(bx)
#         hxy = [hxy; [hxy(end,1) by(j)]; [bx(j)+bdx/2 by(j)]]; %#ok
#     end

#     hxy = [hxy; [hxy(end,1) 0]; hxy(1,:)];
#     fill(hxy(:,1),hxy(:,2),1, ...
#     'EdgeColor',[.6 .0 .0],'FaceColor',[.6 .0 .0],'LineWidth',0.01);
#     hold on;
#     set(gca,'xTick',sort([0:0.2:0.8 th_ml]))
#     set(gca,'xTickLabel',{'0','0.2','0.4','ML','','0.8'})
#     if i == 1; title('SMC^2'); end
#     xlim([min(hist_v)-0.05 max(hist_v)+0.05])
#     xlabel('beta')

#     burn_in = round(min(K(i)/2,500));
#     subplot(3,3,3*i)
#     [by,bx] = hist(theta{i}(1,burn_in:end),hist_v);

#     ind = find(by>0,1,'first'):find(by>0,1,'last');
#     by = by(ind); bx = bx(ind);

#     bdx = diff(bx); bdx = bdx(1);
#     hxy = [bx(1)-bdx/2 0];
#     by = by/(K(i)-burn_in);

#     for j=1:numel(bx)
#         hxy = [hxy; [hxy(end,1) by(j)]; [bx(j)+bdx/2 by(j)]]; %#ok
#     end

#     hxy = [hxy; [hxy(end,1) 0]; hxy(1,:)];
#     h = fill(hxy(:,1),hxy(:,2),1, ...
#     'EdgeColor',[.6 .0 .0],'FaceColor',[.6 .0 .0],'LineWidth',0.01);
#     hold on;
#     set(gca,'xTick',sort([0:0.2:0.8 th_ml]))
#     set(gca,'xTickLabel',{'0','0.2','0.4','ML','','0.8'})
#     xlim([min(hist_v)-0.05 max(hist_v)+0.05])
#     if i == 1; title('PMH'); end
#     xlabel('\beta')
#     axis_v = axis;

#     subplot(3,3,3*i-1)
#     axis(axis_v)

#     subplot(3,3,3*i-2)
#     text(0,0.7,['PMH: K = ', num2str(K(i)), ', N = ', num2str(N(i)), ' (',num2str(pmh_time(i)/60,1),' min)' ])
#     text(0,0.3,['SMC^2: N_ theta = ', num2str(N_th(i)), ', K_m = ', num2str(Km(i)), ' (',num2str(smc2_time(i)/60,1),' min)' ])
#     axis off

# end
# %%
# figure(3), clf
# t_vec = [1 20:20:T];
# N_th_s = 5;
# ap = systematic_resampling(1/N_th(i)*ones(1,N_th(i)),N_th_s);
# for t = t_vec
#     scatter(t*ones(N_th_s,1),x_th{3}(1,ap,t),w_th{3}(ap,t)*10000+1,[0.6 0 0],'filled')
#     hold on
# end
# ylim([-0.5 1.5])
# set(gca,'xTick',t_vec)
# xlabel('Time t')
# ylabel('beta_t')

# %%
# figure(3), clf
# t_vec = [1 20:20:T];
# N_th_s = 50;
# ap = systematic_resampling(1/N_th(i)*ones(1,N_th(i)),N_th_s);
# i = 3;
# for t = t_vec
#     scatter(t*ones(N_th_s,1),x_th{i}(1,ap,t),w_th{i}(ap,t)*10000+1,[0.6 0 0],'filled')
#     hold on
# end
# ylim([-0.5 1.5])
# set(gca,'xTick',t_vec)
# xlabel('Time t')
# ylabel('beta_t')

# %%
# figure(2), clf

# i = 3;
# plot(theta{i}(1:min(2000,K(i))),'linewidth',1,'color',[0.6 0 0])
# set(gca,'xTick',0:1000:2000)
# xlabel('Iteration k')
# ylabel('beta[k]')
indx = zeros(1, N);
