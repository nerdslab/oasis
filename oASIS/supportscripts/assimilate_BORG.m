% oASIS: Adaptive Column Sampling for Kernel Matrix Approximation
% R. Patel, T. Goldstein, E. Dyer, A. Mirhoseini, and R. Baraniuk
% Submitted to IEEE JSTSP
%
% Support Script to make a dataset.
%
% assimilate_BORG generates points clustered with variance 0.1 
% at the vertices of an n-dimensional unit cube.
% 
% Inputs:
%   
%       ndim                    -   number of dimensions of the hypercube.
%       numpoints_per_cluster   -   self explanatory
%
% Output:
%
%       X, the matrix containing all of the datapoints arranged columnwise.
%
%
%


function [X] = assimilate_BORG(ndim,numpoints_per_cluster)

% First, make a huge cluster of gaussian points with variance 0.1
X = randn(numpoints_per_cluster*2^ndim, ndim);
X = bsxfun(@times,.1/(ndim)^.5,X);

% Then, divide those points equally around each vertex.
for vertex = 1:2^ndim-1
    t = dec2bin(vertex);
    t = t(end:-1:1);
    corner_coords = find(t == '1');
    
    indices = [numpoints_per_cluster*(vertex-1)+1:numpoints_per_cluster*(vertex)];
    
    
    X(indices,corner_coords) = X(indices,corner_coords) + 1;
end

% Arrange the data columnwise.
X = X';

return