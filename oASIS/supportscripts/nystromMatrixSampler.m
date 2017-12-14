% oASIS: Adaptive Column Sampling for Kernel Matrix Approximation
% R. Patel, T. Goldstein, E. Dyer, A. Mirhoseini, and R. Baraniuk
% Submitted to IEEE JSTSP
%
% Support Script to help compute approximation errors by computing points
% from the approximation from C and W+, in large datasets where computing
% the entire kernel matrix is infeasible.
%
%


function [ rval ] = nystromMatrixSampler( C,CWplus, r, c)

% return list of entries of approximation.
if length(r)>1
    rList = CWplus(c,:);
    cList = C(r,:);
    rval = sum(rList.*cList,2);
    return;
end

% return diagonal of approximation.
if strcmp(r,'D') || strcmp(c,'D')
    rval = sum(C.*CWplus,2);
    return;
end

% return single entry of approximation.
if ~isempty(r)
    rval = CWplus(r,:)*C(c,:)';
    
    return;
end
% return column of approximation.
rval = CWplus*C(c,:)';
return;

end

