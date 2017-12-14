% oASIS: Adaptive Column Sampling for Kernel Matrix Approximation
% R. Patel, T. Goldstein, E. Dyer, A. Mirhoseini, and R. Baraniuk
% Submitted to IEEE JSTSP
%
% Support Script as an example of a function handle for an implicit kernel
% matrix. This handle creates Gaussian Kernel Matrix entries.
%
%  "Z" is a dataset with one data point in each column.
%  
%  GaussianKernelMatrixSampler(Z,r,c,sigma) returns
%          exp(-d(r,c)/sigma^2),
%  where d(r,c) denotes the distance between the point at column "r" and
%  column "c." 
%
%  GaussianKernelMatrixSampler(Z,[],c,sigma) returns 
%  the entire column "c" from the similarity matrix.  
%
%  GaussianKernelMatrixSampler(Z,'D','D',sigma) returns a diagonal.


function [ rval ] = GaussianKernelMatrixSampler( Z, r, c, sigma )

% If we're given a list of points, return the gaussian distances
if length(r)>1
    rList = Z(:,r)';
    cList = Z(:,c)';
    diff = rList-cList;
    diff = diff.*diff;
    rval = exp(-sum(diff,2)/sigma/sigma/2);
    return;
end

% The Diagonal of a Gaussian Kernel matrix is all ones.
if strcmp(r,'D') || strcmp(c,'D')
    rval = ones(size(Z,2),1);
    return;
end

% Compute a single point in the Gaussian matrix.
if ~isempty(r)
    rval = exp(-norm(Z(:,r) - Z(:,c),'fro').^2/sigma/sigma/2);
    return;
end

% Compute a column of the Gaussian matrix at c.
d2 = sum(bsxfun(@minus,Z,Z(:,c)).^2,1);
rval = exp(-d2/sigma/sigma/2)';
return;

end
