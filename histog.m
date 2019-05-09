function histo = histog(X,beta, binedges)
    h = histogram(X,binedges);
    h_val = h.Values;
    w = h_val/sum(h_val);
    h_bins = h.BinEdges;
    kappa = zeros(1,length(h_bins)-1);
    for i = 1:length(kappa)
        kappa(1,i) = 1/(h_bins(i+1)-h_bins(i));
    end
    size(beta)
    size(w)
    size(kappa)
    pdf = beta.*w.*kappa;
    sum(beta.*w)
    pdf = pdf/ sum(beta.*w);
    %pdf
    %beta
    figure
    histogram('BinEdges',h_bins,'BinCount',pdf)
end