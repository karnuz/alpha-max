n_c = 2;
%X1 = unbpdata(1000);
X1 = bias_pos_data();
X = mix_data();

opts.no_of_clusters = n_c;
opts.consType='eq';
opts.num_restarts = 1;
opts.constraints=0.01:0.01:0.99;

%learnbeta=init_learnbeta_zeta(X,X1,[],[],opts);

%opts.transform=@(X,X1)(transform_nn(X,X1,struct('h',7)));
[est, out] = estimateMixprop(X, X1,'AlphaMax_B',opts);
%[est1, out1] = estimateMixprop(X, X1,'AlphaMax',opts);


function data = bias_pos_data()
%    data = [rand(700,1);-5+rand(300,1)];
    data = [randn(700,1);-5+randn(300,1)];
end

function data = mix_data()
%    data = [-5+rand(1000,1);rand(1000,1);10+randn(8000,1)];
    data = [-5+randn(1000,1);randn(1000,1);10+randn(8000,1)];
end


%unbiased positive array
function uparray = unbpdata(size,mean,std)
    uparray = mean + std*randn(size,1);
end

%unbiased unlabelled array
function unbularray = unbuldata(size, alpha)
    M = round(size*alpha);
    N = size - M;
    unbularray = [ndata(N); unbpdata(M,0,3)];
end

% biased unlabelled data
function uarray = udata(size, alpha)
    M = round(size*alpha);
    N = size - M;
    uarray = [ndata(N); bpdata(M)];
end

%negative data
function narray = ndata(size)
    narray = randn(size,1) + 10;
end


%biased positive data
function parray = bpdata(size,k)
    mean = 0;
    std = 3;
    stdbump = std/10;
    N = size;
    K = k;
    gamma = 0.7;

    kbumpmeans = mean-3*std + 6*std*rand(1,K);
    parray = zeros(N,1);

    for n = 1:N
        z = coin(gamma);
        if z == 0
            parray(n) = unbpdata(1,mean,std);
        else
            idx = randi([1,K],1,1);
            parray(n,1) = kbumpmeans(idx) + stdbump*randn(1,1);
        end
    end
end


function y = coin(x)
    z = rand(1,1);
    if z < x
        y = 0;
    else
        y = 1;
    end
end


function [ cons ] = logscale_cons(lower_limit,upper_limit, num_cons)
%generates num_cons values between lower_limit and upper_limit. The values
%are equally spaced in log scale.
l_log=log(lower_limit);
u_log=log(upper_limit);
sep=(u_log-l_log)/num_cons;
cons=exp(l_log:sep:u_log);
cons=cons(1:end-1);
end