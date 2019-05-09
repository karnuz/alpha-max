function [learnbeta] = init_learnbeta_zeta(mixture_sample,component_sample,p,p1,opts)

    loss_fcn = @combined_loss;
    
    n_c = opts.no_of_clusters;
    [idx, c_comp] = kmeans(component_sample, n_c);
    c_comp = sort(c_comp);
    
    comp_cluster_boundaries = zeros(1,n_c+1);

    %find cluster boundaries
    for n = 1:length(c_comp)-1
        comp_cluster_boundaries(n+1) = (c_comp(n) + c_comp(n+1))/2;
    end
    comp_cluster_boundaries(1) = min(component_sample);
    comp_cluster_boundaries(n_c+1) = max(component_sample);
    
%    comp_cluster_boundaries(1) = c_comp(1) - ((c_comp(2)-c_comp(1))/2);
%    comp_cluster_boundaries(n_c+1) = c_comp(n_c) + ((c_comp(n_c)-c_comp(n_c-1))/2);

    
    %histogram object
    comp_hist = histogram(component_sample,100);
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
    mix_hist_values = mix_hist.Values;
    w_mix = mix_hist.Values/(sum(mix_hist.Values));
    mix_bin_edges = mix_hist.BinEdges;
    numkernels = length(mix_hist.Values);

    options = optimoptions('fmincon','GradObj','on', 'Display','off');
    learnbeta = @learnbeta_fcn;
    
    
    %here beta_init should have beta followed by zeta
    function [beta_zeta,f,alpha,iter] = learnbeta_fcn(alpha_cons,beta_init)
        %minimizes the negative log likelihood.
        if nargin < 1
            alpha_cons=0;
        end
        opts.A = [-1*eye(numkernels+n_c);eye(numkernels+n_c)];
        opts.b = [zeros(numkernels+n_c,1);ones(numkernels+n_c,1)];
        % fmin_BFGS enforces Ax >= b, rather than fmincons Ax <= b
        opts.A = -1*opts.A;
        opts.b = -1*opts.b;
        alphas = zeros(1,opts.num_restarts);
        fs = zeros(1,opts.num_restarts);
        iters=zeros(1,opts.num_restarts);
        betas_zetas=zeros(numkernels+n_c,opts.num_restarts);
        for rr= 1:opts.num_restarts
            if strcmp(opts.consType,'ineq')
                opts.A = [opts.A;avec'];
                opts.b = [opts.b;alpha_cons];
                if nargin<2
                    beta_init = rand(numkernels,1);
                    beta_init = alpha_cons + (1-alpha_cons-10^-7)*beta_init;
                end
            elseif strcmp(opts.consType,'eq')
                opts.Aeq=[w_mix,zeros(1,n_c);zeros(1,length(w_mix)),ones(1,n_c)];
                opts.beq=[alpha_cons;1];
                %if nargin < 2
                beta_init =repmat(alpha_cons,numkernels,1);
                beta_zeta_init = [beta_init;repmat(1/n_c,n_c,1)];
                %end
            end
            [beta_zeta_i,f_i,iter_i,flag] = fmincon_caller(loss_fcn,beta_zeta_init, opts);
            %[beta_i,f_i,iter_i,flag] = fmin_LBFGS(loss_fcn,beta_init, opts);
            betas_zetas(:,rr) = beta_zeta_i;
            alphas(rr)=w_mix*beta_zeta_i(1:numkernels);
            fs(rr)=f_i;
            iters(rr)=iter_i;
        end
        [f,ix_min]=min(fs);
        beta_zeta=betas_zetas(:,ix_min);
        alpha=alphas(ix_min);
        iter=iters(ix_min);
    end

    function [f,g] = combined_loss(beta_zeta)
        
        beta = (beta_zeta(1:numkernels))';
        zeta = (beta_zeta(numkernels+1:end))';
        
        alpha = sum(w_mix.*beta);

        binedges = mix_hist.BinEdges;
        kappa = zeros(1,length(binedges)-1);
        for i = 1:length(kappa)
            kappa(1,i) = 1/(binedges(i+1)-binedges(i));
        end

        h_0 = (1-beta).*w_mix.*kappa;

        comp_binedges = comp_bin_edges;
        kappa_comp = zeros(1,length(comp_binedges)-1);
        for i = 1:length(kappa_comp)
            kappa_comp(1,i) = 1/(comp_binedges(i+1)-comp_binedges(i));
        end

        zeta_array = zeros(1,length(comp_hist_values));
        for i = 1:length(cluster_boundary_index)-1
            zeta_array(cluster_boundary_index(i):cluster_boundary_index(i+1)-1) = zeta(i);
        end

        f_1 = zeta_array.*w_comp.*kappa_comp;

        min_comp = min(comp_bin_edges);
        idx = length(bins1)+1;
%        idx = find_nearest(min_comp, mix_hist.BinEdges);

        loss_arr = h_0;
        loss_arr(idx:idx+length(f_1)-1) = loss_arr(idx:idx+length(f_1)-1) + alpha*f_1;

        log_loss_arr = log(loss_arr);
        log_loss_arr(log_loss_arr==-Inf) = 0;

        f = -sum(mix_hist_values.*log_loss_arr);
        
        %calculate g now
%        h_beta_der = -mix_hist_values.*w_mix.*kappa;
%        f_beta_der = h_beta_der ./ mix_hist_values.*log_loss_arr;

        h_beta_der = -w_mix.*kappa;
        f_beta_der = h_beta_der./loss_arr;
        f_beta_der(isnan(f_beta_der)) = 0;
        f_beta_der = f_beta_der.*mix_hist_values;

        f_zeta_der = zeros(1,n_c);
        comp_pdf = alpha*(w_comp.*kappa_comp);
        for i = 1:n_c
            f_zeta_der(i) = sum(comp_hist_values(cluster_boundary_index(i):cluster_boundary_index(i+1)-1).*comp_pdf(cluster_boundary_index(i):cluster_boundary_index(i+1)-1));
        end
        f_beta_zeta_der = [f_beta_der,f_zeta_der];
        g = -f_beta_zeta_der;
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

    function [beta,f,iter,flag]=fmincon_caller(loss_fcn,beta_init,opts)
        if strcmp(opts.consType,'eq')
            [beta,f,flag,output] = fmincon(loss_fcn,beta_init, -opts.A,-opts.b,opts.Aeq,opts.beq,[],[],[],options);
        elseif strcmp(opts.consType,'ineq')
            [beta,f,flag,output] = fmincon(loss_fcn,beta_init, -opts.A,-opts.b,[],[],[],options);
        end
         iter=output.iterations;
    end
    
end


