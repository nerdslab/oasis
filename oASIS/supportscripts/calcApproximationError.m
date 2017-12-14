% oASIS: Adaptive Column Sampling for Kernel Matrix Approximation
% R. Patel, T. Goldstein, E. Dyer, A. Mirhoseini, and R. Baraniuk
% Submitted to IEEE JSTSP
%
% Support Script to compute relative error between the kernel matrix and
% the approximation, when the dataset is large enough to make an
% explicit kernel matrix infeasible
% 
% Inputs:
%   
%       f      -   Kernel matrix or function handle to kernel matrix.
%       outs   -   output structure of oASIS. May be "nys" in demo.
%
% Output:
%
%       relerror -  Error from 100,000 random points.
%
%
%

function [relError] = calcApproximationError( f,outs)

N = length(f([],1)); %need to know number of points in matrix


if isnumeric(f) %exact matrix
    relError = norm(outs.Gtilde(:)-f(:),'fro')/norm(f(:),'fro');
else
    
    numSamples = 100000;
    a = zeros(numSamples,1);
    b = zeros(numSamples,1);
    CWplus = outs.C*outs.Wplus;
    
    %Note that our approximation would also be very large, so we need a
    %function handle similar to the actual kernel matrix
    ny = @(r,c) nystromMatrixSampler(outs.C,CWplus,r,c);
    rng(547,'twister');
    
    %%Added a loop here because 100000 samples was too large for computer
    for tempi = 1:(numSamples/10000)
        rows = randi(N,10000,1);
        cols = randi(N,10000,1);
        a((tempi-1)*10000+1:tempi*10000) = f(rows,cols);
        b((tempi-1)*10000+1:tempi*10000) = ny(rows,cols);    
    end
    relError = norm(a-b,'fro')/norm(a,'fro');
end