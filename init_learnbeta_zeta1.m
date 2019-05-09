function [learnbeta] = init_learnbeta_zeta1(mix_hist,comp_hist,cluster_boundary_index,opts)

    loss_fcn = @combined_loss;
    n_c = opts.no_of_clusters;
    comp_start_idx = opts.comp_start_idx;
    
    %histogram object
    w_comp = comp_hist.w_comp;
    comp_hist_values = comp_hist.Values;
    comp_bin_edges = comp_hist.BinEdges;

%now we have histogram of compoenent and the cluster boundary indices.
    
    mix_hist_values = mix_hist.Values;
    w_mix = mix_hist.w_mix;
    mix_bin_edges = mix_hist.BinEdges;
    numkernels = length(mix_hist.Values);

    kappa_comp = zeros(1,length(comp_bin_edges)-1);
    for j = 1:length(kappa_comp)
        kappa_comp(1,j) = 1/(comp_bin_edges(j+1)-comp_bin_edges(j));
    end
    
    binedges = mix_bin_edges;
    kappa = zeros(1,length(binedges)-1);
    for j = 1:length(kappa)
        kappa(1,j) = 1/(binedges(j+1)-binedges(j));
    end
    
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

        h_0 = (1-beta).*w_mix.*kappa;

        zeta_array = zeros(1,length(comp_hist_values));
        for i = 1:length(cluster_boundary_index)-1
            zeta_array(cluster_boundary_index(i):cluster_boundary_index(i+1)-1) = zeta(i);
        end

        f_1 = zeta_array.*w_comp.*kappa_comp;

        min_comp = min(comp_bin_edges);
        idx = comp_start_idx;
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
            der = comp_pdf(cluster_boundary_index(i):cluster_boundary_index(i+1)-1)./loss_arr(idx+cluster_boundary_index(i)-1:idx+cluster_boundary_index(i+1)-2);
            der(isnan(der)) = 0;
            f_zeta_der(i) = sum(mix_hist_values(idx+cluster_boundary_index(i)-1:idx+cluster_boundary_index(i+1)-2).*der);
        end
        f_beta_zeta_der = [f_beta_der,f_zeta_der];
        g = -f_beta_zeta_der;
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


