arr = [1,2,3,4,5,6,7,8,9,10];
find_nearest(6,arr,1,length(arr))


function idx = find_nearest(val, arr,l,u)
    if l==u
        idx = l;
        return
    end

    m = floor((l+u)/2);
    if arr(m)<= val & val <= arr(m+1)
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


function [f,g] = combined_loss(beta_zeta)

    beta = (beta_zeta(1:numkernels))';
    zeta = (beta_zeta(numkernels+1:end))';

    alpha = sum(w_mix.*beta);

    binedges = mix_hist.BinEdges;
    kappa = zeros(1,length(binedges)-1);
    for i = 1:len(kappa)
        kappa(1,i) = 1/(binedges(i+1)-binedges(i));
    end

    h_0 = (1-beta).*w_mix.*kappa;

    comp_binedges = comp_hist.BinEdges;
    kappa_comp = zeros(1,length(comp_binedges)-1);
    for i = 1:len(kappa_comp)
        kappa_comp(1,i) = 1/(comp_binedges(i+1)-comp_binedges(i));
    end

    zeta_array = zeros(1,length(comp_hist.Values));
    for i = 1:length(cluster_boundary_index)-1
        zeta_array(cluster_boundary_index(i):cluster_boundary_index(i+1)-1) = zeta(i);
    end

    f_1 = zeta_array.*w_comp.*kappa_comp;

    min_comp = min(comp_hist.BinEdges);
    idx = length(bins1)+1;
%        idx = find_nearest(min_comp, mix_hist.BinEdges);

    loss_arr = h_0;
    loss_arr(idx:idx+length(f_1)-1) = loss_arr(idx:idx+length(f_1)-1) + alpha*f_1;

    log_loss_arr = log(loss_arr);

    f = sum(mix_hist.Values.*log_loss_arr);

    %calculate g now
    h_beta_der = -mix_hist.Values.*w_mix.*kappa;
    f_beta_der = h_beta_der ./ mix_hist.Values.*log_loss_arr;

    f_zeta_der = zeros(1,numclusters);
    comp_pdf = alpha*(w_comp.*kappa_comp);
    for i = 1:numclusters
        f_zeta_der(i) = sum(comp_hist.Values(cluster_boundary(i):cluster_boundary(i+1)-1).*comp_pdf(cluster_boundary(i):cluster_boundary(i+1)-1));
    end
    f_beta_zeta_der = [f_beta_der,f_zeta_der];
    g = f_beta_zeta_der;
end
