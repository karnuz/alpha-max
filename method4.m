X1 = bpdata(1000);
X = unbuldata(10000,0.5);
opts.transform=@(X,X1)(transform_nn(X,X1,struct('h',7)));
[est, out] = estimateMixprop(X, X1,'AlphaMax',opts);


%unbiased positive array
function uparray = unbpdata(size)
    uparray = 2*randn(size,1);
end

%unbiased unlabelled array
function unbularray = unbuldata(size, alpha)
    M = round(size*alpha);
    N = size - M;
    unbularray = [ndata(N); unbpdata(M)];
end

% biased unlabelled data
function uarray = udata(size, alpha)
    M = round(size*alpha);
    N = size - M;
    uarray = [ndata(N); bpdata(M)];
end

%negative data
function narray = ndata(size)
    narray = 2*randn(size,1) + 10;
end


%biased positive data
function parray = bpdata(size)
    mean = 0;
    std = 3;
    stdbump = std/10;
    N = size;
    K = 10;
    gamma = 0.7;

    kbumpmeans = mean-3*std + 6*std*rand(1,K);
    parray = zeros(N,1);

    for n = 1:N
        z = coin(gamma);
        if z == 0
            parray(n) = mean+ std*randn(1,1);
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