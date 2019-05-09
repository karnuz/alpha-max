function [alphas, fs, out] = compute_llCurve_B1(x,x1,opts)
%computes the log likelihood versus alpha(mixing proportion)
DEF.densityEst_fcn=@densEst_hist;
DEF.constraints = 0.01:0.01:0.99;
DEF.consType='eq';
DEF.parallel=false;
DEF.num_restarts=1;
DEF.gamma=1/length(x);
DEF.gamma1=1/length(x1);
DEF.loss_str='combined';
if nargin < 3
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end
if ~isfield(opts,'dens')
    [dens.p,dens.p1]=opts.densityEst_fcn(x,x1);
else
    dens=opts.dens;
end
loss_strs={'combined','combined2','component','mixture'};
ll_strs={'ll_cmb','ll_cmb2','ll_cmp','ll_mix'};
str=validatestring(opts.loss_str,loss_strs);
jx=find(strcmp(str,loss_strs));
ll_str=ll_strs{jx};

out.dens=dens;
numkernels=length(dens.p.mixProp);
opts1.num_restarts=opts.num_restarts;
opts1.lossfcn=opts.loss_str;
opts1.consType=opts.consType;
opts1.gamma=opts.gamma;
opts1.gamma1=opts.gamma1;
opts1.no_of_clusters = opts.no_of_clusters;

%him

n_c = opts.no_of_clusters;
component_sample = x1;
mixture_sample = x;
[idx, c_comp] = kmeans(component_sample, n_c);
c_comp = sort(c_comp);

comp_cluster_boundaries = zeros(1,n_c+1);

%find cluster boundaries
for n = 1:length(c_comp)-1
    comp_cluster_boundaries(n+1) = (c_comp(n) + c_comp(n+1))/2;
end
comp_cluster_boundaries(1) = min(component_sample);
comp_cluster_boundaries(n_c+1) = max(component_sample);


%histogram object
comp_hist = histogram(component_sample);
w_comp = comp_hist.Values/(sum(comp_hist.Values));
comp_hist_values = comp_hist.Values;
comp_bin_edges = comp_hist.BinEdges;

cluster_boundary_index = zeros(1,n_c+1);
cluster_boundary_index(1) = 1;
cluster_boundary_index(n_c+1) = length(comp_hist.BinEdges);
for j = 2:n_c
    cluster_boundary_index(j) = find_nearest(comp_cluster_boundaries(j),comp_hist.BinEdges,1,length(comp_hist.BinEdges));
end

for j = 1:length(cluster_boundary_index)-1
    w_comp(cluster_boundary_index(j):cluster_boundary_index(j+1)-1) = w_comp(cluster_boundary_index(j):cluster_boundary_index(j+1)-1)/sum(w_comp(cluster_boundary_index(j):cluster_boundary_index(j+1)-1));
end

comp_histo.Values = comp_hist.Values;
comp_histo.BinEdges = comp_hist.BinEdges;
comp_histo.w_comp = w_comp;

kappa_comp = zeros(1,length(comp_bin_edges)-1);
for i = 1:length(kappa_comp)
    kappa_comp(1,i) = 1/(comp_bin_edges(i+1)-comp_bin_edges(i));
end

%now we have histogram of compoenent and the cluster boundary indices.

min_mix = min(mixture_sample);
max_mix = max(mixture_sample);
comp_min_hist = comp_hist.BinEdges(1);
comp_max_hist = comp_hist.BinEdges(end);
stepsize = comp_hist.BinWidth;

%    bins1 = [];
%    if min_mix < comp_min_hist
%        bins1 = comp_min_hist-stepsize:-stepsize:min_mix-stepsize;
%        bins1 = fliplr(bins1);
%    end
%    bins2 = [];
%    if max_mix > comp_max_hist
%        bins2 = comp_max_hist+stepsize:stepsize:max_mix+stepsize;
%    end


bins1 = [];
if min_mix < comp_min_hist
    bins1 = min_mix;
end
bins2 = [];
if max_mix > comp_max_hist
    bins2 = max_mix;
end

mix_binedges = [bins1,comp_hist.BinEdges,bins2];
mix_hist = histogram(mixture_sample,mix_binedges);
w_mix = mix_hist.Values/(sum(mix_hist.Values));
mix_bin_edges = mix_hist.BinEdges;
numkernels = length(mix_hist.Values);

mix_histo.Values = mix_hist.Values;
mix_histo.BinEdges = mix_hist.BinEdges;
mix_histo.w_mix = w_mix;
opts1.comp_start_idx = length(bins1)+1;

kappa = zeros(1,length(mix_bin_edges)-1);
for i = 1:length(kappa)
    kappa(1,i) = 1/(mix_bin_edges(i+1)-mix_bin_edges(i));
end

%him_end

learnbeta=init_learnbeta_zeta1(mix_histo,comp_histo,cluster_boundary_index,opts1);

cons=opts.constraints;
num_lbs=length(cons);
alphas=zeros(1,num_lbs);
ll_cmb=nan(1,num_lbs);
ll_cmp=nan(1,num_lbs);
ll_mix=nan(1,num_lbs);
ll_cmb2=nan(1,num_lbs);
fs=nan(1,num_lbs);
objs=nan(1,num_lbs);
iters=zeros(1,num_lbs);
betaszetas=nan;
betas=nan;
zetas=nan;

%betas=nan(numkernels+opts.no_of_clusters,num_lbs);
init=nan(numkernels,num_lbs);
ll_bt= @(beta)ll_beta(beta,x,x1,dens.p,dens.p1,opts);
if(opts.parallel)
    parfor k = 1:num_lbs
        [beta,o,alpha,iter] = feval(learnbeta,cons(k));
        betas(:,k) =beta;
        if(isnan(o))
            error('objective nan');
        end
        alphas(k)=alpha;
        objs(k)=o;
        [ll_cmb(k),ll_mix(k),ll_cmp(k),ll_cmb2(k)]=ll_bt(beta);
        iters(k)=iter;
    end
else
    for k = 1:num_lbs
        
        beta_init =get_beta_init(k);
        if any(isnan(beta_init))
            [betazeta,o,alpha,iter] = feval(learnbeta,cons(k));
        else
            try
                   [betazeta,o,alpha,iter] = feval(learnbeta,cons(k));
            catch
                warning('optimization failed')
            end
        end
        if(isnan(o))
            error('objective nan');
        end
        if k==1
            betaszetas=nan(length(betazeta),num_lbs);
            betas = nan(length(betazeta)-opts.no_of_clusters,num_lbs);
            zetas = nan(opts.no_of_clusters,num_lbs);
        end

        betaszetas(:,k) = betazeta;
        betas(:,k) = betazeta(1:length(betazeta)-opts.no_of_clusters);
        zetas(:,k) = betazeta(length(betazeta)-opts.no_of_clusters+1:end);
        alphas(k)=alpha;
        objs(k)=o;
        %[ll_cmb(k),ll_mix(k),ll_cmp(k),ll_cmb2(k)]=ll_bt(betas(:,k));
        iters(k)=iter;
%        update_init(k,betas(:,k));
    end
end
fs = -objs;

ys = zeros(size(fs));
for i = 1:length(fs)
%    t = betas(1:length(bins1),i).*w_mix()
    t = betas(length(bins1)+1:length(bins1)+length(w_comp),i)'.*w_mix(length(bins1)+1:length(bins1)+length(w_comp)).*kappa_comp;
    t = t/alphas(i);
    t = log(t);
    t(t==-Inf) = 0;
    ys(i) = sum(comp_hist_values.*t);
end


%fs = fs/10000 + ys/1000;



%fs=eval(ll_str);
out.objs=objs;
out.zetas = zetas;
out.betas=betas;
out.iters=iters;
out.ll_cmb=ll_cmb;out.ll_mix=ll_mix;out.ll_cmp=ll_cmp;out.ll_cmb2=ll_cmb2;
out.alphas=alphas;
out.fs=fs;
out.cluster_boundaries = cluster_boundary_index;
out.mix_bin_edges = mix_bin_edges;
out.comp_bin_edges = comp_bin_edges;


    function beta_init=get_beta_init(kk)
        beta_init=init(:,kk);
    end
    function update_init(kk,bt)
        if kk < num_lbs
            bt=min(1-10^-8,bt);
            bt=max(10^-8,bt);
            if strcmp(opts.consType,'eq')
                while abs(cons(kk+1)-sum(bt.*dens.p.mixProp')) >1e-12;
                    c=cons(kk+1)/sum(bt.*dens.p.mixProp');
                    bt=min(1-10^-8,c*bt);
                end
            elseif strcmp(opts.consType,'ineq')
                bt=min(1-10^-8,bt);
                bt=max(10^-8,bt);
                while sum(bt.*dens.p.mixProp')-cons(kk+1) < 1e-4
                    c=(cons(kk+1)+1e-3)/sum(bt.*dens.p.mixProp');
                    bt=min(1-10^-8,c*bt);
                end
            end
            init(:,kk+1)=bt;
        end
    end

    function idx = find_nearest(val, arr,l,u)
        if l==u
            idx = l;
            return
        end

        m = floor((l+u)/2);
        if arr(m)<= val && val <= arr(m+1)
            if abs(val-arr(m)) >= abs(val-arr(m+1))
                idx = m+1;
            else
                idx = m;
            end
        elseif val > arr(m+1)
            idx = find_nearest(val,arr,m+1,u);
        elseif val < arr(m)
            idx = find_nearest(val, arr,l,m);
        end
    end


end

