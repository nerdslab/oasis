% oASIS: Adaptive Column Sampling for Kernel Matrix Approximation
% R. Patel, T. Goldstein, E. Dyer, A. Mirhoseini, and R. Baraniuk
% Submitted to IEEE JSTSP
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% The MIT License (MIT)
%
% Copyright (c) 2015 Rice University
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% oASIS samples columns for the Nystrom approximation. The Nystrom 
% approximation of a matrix X = C*W^+*C^T, where C is a submatrix of 
% sampled columns from X and W is a submatrix of C at the row indices 
% used when sampling C.
%
%
% USAGE:
%           [ outs ] = oASIS( X, vecOfSampleSizes, method, opts )
%   Inputs:
%       X - a symmetric psd matrix, OR a function handle to a dataset Z
%       and a kernel function. See GaussianKernelMatrixSampler.m in the
%       supportscripts folder for an example of a Gaussian Kernel.
%       
%       vecOfSampleSizes - The number of columns to select. This can be a
%       single value, or a vector. This function will compute C and W for
%       all of the sample sizes in this vector.
%       
%       method - either 'oASIS', or 'random' for an example comparison.
%
%       opts - a structure where you can set various parameters, as
%       detailed below:
% 
%           opts.verbose              To see progress dots and runtime
%           opts.expandNystrom        To return full approximation matrix,
%                                       not just W^+ and C
%           opts.computeApproxSVD     Compute singular values, vectors
%           opts.selection            Explicit indices, no other sampling 
%                                       will be done
%           opts.use_mex              Use a MEX interface for even faster
%                                       performance.
%           opts.use_randomseed       Set to "true" to set the random seed
%                                       before running.
%           opts.startSize            Number of random columns with which 
%                                       to start.
%
%   Outputs:
%       outs - a structure consisting of:
%           C                         Full sampled columns
%           W                         Sampled rows at the column indices
%           Wplus                     Pseudoinverse of W
%           selection                 Vector of selected indices
%           Gtilde (optional)         Full approximation matrix
%           nystroms (optional)       C,W, and Gtilde at all L in 
%                                       vecOfSampleSizes
%           U (optional)              Singular vectors of X
%           S (optional)              Singular values of X
%
%
%   Demos:
%       oASIS_demo_explicit_kernel_matrix.m will run oASIS with a fully
%       precomputed kernel matrix G.
%
%       oASIS_demo_implicit_kernel_matrix.m will run oASIS with a function
%       handle containing a dataset Z and the gaussian kernel.


function [ outs ] = oASIS( X, vecOfSampleSizes, method, opts )
addpath('supportscripts')

%% Check preconditions, fill missing optional entries in 'opts'
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end
opts = fillDefaultOptions(opts);

L = max(vecOfSampleSizes); %  maximum number of cols to sample

assert(strcmp(lower(method),'oasis') | strcmp(lower(method),'o') |  ...
        strcmp(lower(method),'r') | strcmp(lower(method),'random'), ...
        'Method must be either random or oasis.' )

if isnumeric(X) % if we are given a full matrix
    [rows,cols] = size(X);
    assert(rows==cols,'Input X must be square');
    assert(L<=cols,'Cannot have more samples than columns');
    N = cols;
end

if opts.verbose
    fprintf('Nystrom: Sampling Method = %s, L = %s\n', ...
        method, num2str(vecOfSampleSizes));
    tic;
else
    tStart = tic;
end


%%  to hold outputs
outs = [];

%%  Sample columns using the method chosen by the user
C = [];
if isempty(opts.selection)
    switch lower(method)
        case {'o','oasis'}
            [selection, C] = innerProductSample_fast(X,L,opts.startSize,opts.verbose,opts.use_mex, opts.use_randomseed);
        case {'r','random'}
            [selection, C]= randomSample(X,L,opts.verbose, opts.use_randomseed);
        otherwise
            error(['Invalid method: ' method]);
    end
else
    selection = opts.selection;  % Or use the columns specified by the user
end

if ~opts.verbose
    outs.runtime = toc(tStart);
end

assert(L<=numel(selection),'L is too big.  Not that many columns have been selected.');


%%  Get C
if isempty(C)
    if isnumeric(X)
        C = X(:,selection);
    else
        N = length(X('D','D'));  %  Problem size
        C = zeros(N,length(selection));
        for i=1:length(selection)
            C(:,i) = X([],selection(i));
        end
    end
else
    N = size(C,1);
end


%% Get W from C
%  Note:  We do not permute the rows of C.  This is so that the matrix we 
%  get back is an approximation of X, not of the permuted X

W = C(selection,:);

%%  Record Results

outs.C = C;
outs.W = W;
outs.selection = selection;

%%  Compute pseudoinverse of W
tic;
if ~issparse(W)
    [Uw,Sw,Vw] = svd(W);
else
    [Uw,Sw,Vw] = svd(full(W));
end
Splus = zeros(size(Sw));
Splus(Sw>1e-7) = Sw(Sw>1e-7).^-1; % invert singular values of W
outs.Wplus = Uw*Splus*Vw';   % The pseudo-inverse of W

if opts.verbose; fprintf('\tfactor: time = %d sec\n',toc);end;



%% Everything below this line only gets called when "opts" is set properly

%% Expand Nystrom Approximation
if opts.expandNystrom
    outs.Gtilde = C*outs.Wplus*C';
end


%%  Compute psuedo-inverse of W
%Do this explicitly (rather than calling 'pinv') and store SVD
if opts.computeApproxSVD
    % Compute approximate singular values of X
    outs.S = (N/L)*diag(Sw);
    % Compute approximate singular vectors of X
    outs.U = sqrt(L/N)*(C*Uw*Splus);
end


%%  If more than one sample size was chosen, compute them all
if length(vecOfSampleSizes)>1
    outs.nystroms = {};
    for Lsmall = vecOfSampleSizes
        thisNystrom = [];
        thisNystrom.C = C(:,1:Lsmall);
        thisNystrom.W = thisNystrom.C( selection(1:Lsmall),:);
        thisNystrom.L = Lsmall;
        if opts.expandNystrom
            thisNystrom.Gtilde = thisNystrom.C*(thisNystrom.W\thisNystrom.C');
        end
        outs.nystroms(end+1) = {thisNystrom};
    end
end


return



%% oASIS with C/MEX interface, only for precomputed X
function [indices, C ] = innerProductSample_fast(X,L, startSize, verbose, use_mex, use_randomseed)
if use_randomseed; rng(0,'twister'); end
if verbose; fprintf('\tBegin oASIS\n'); tic; end
if isnumeric(X) && use_mex
    if ~exist('asis')
        fprintf('Compiling asis executable.  You may get an error if you have not setup mex.\n');
        mex supportscripts/asis.c;
    end
    indices = asis(X,L, double(verbose));
    indices = indices+1; %  Convert from C 0-indexed array to matlab 1-indexing
    C = [];
else
    if verbose;fprintf('\t\t\t');end
    [indices, C] = innerProductSample_dense(X,L, startSize, verbose, use_randomseed);
end
if verbose; fprintf('\n\tEnd Product: time = %f secs\n',toc); end
return



%% oASIS with MATLAB, for data and a kernel or a precomputed X
function [indices, C ] = innerProductSample_dense(X,L, startSize, verbose, use_randomseed)
if use_randomseed; rng(0,'twister'); end

%% Get the diagonal
if ~isnumeric(X)
    D = X('D','D')';
else
    D = diag(X)';
end
N = length(D);  %  Problem size

%% Randomly choose some starting columns
randomColPerm = randperm(N);
startSize = min(startSize,L);
indices = randomColPerm(1:startSize);  %  the indices of selected columns
C = [];
if ~isnumeric(X)
    for ii = 1:length(indices)
        newcol = X([],indices(ii));
        C = [C newcol];
    end
else
    C = X(:,indices);
end

%  the selected columns themselves
W = C(indices,:);
R = W\C';
delta = (sum(R.*C',1)-D);
[m,newColIndex] = max(abs(delta));

while size(R,1)<L
    %% Correct the matrix R = W^{-1}*C' to add columns we've already selected
    if ~isnumeric(X)
        newCol = X([],newColIndex);
    else
        newCol = X(:,newColIndex);
    end
    b = newCol(indices);
    d = newCol(newColIndex);
    Ainvb = R(:,newColIndex);
    shur = (d-b'*Ainvb)^-1;
    
    %% Use block matrix inverse formula to add row to W^{-1}*C
    brep = b'*R;
    R = [R+Ainvb*shur*(brep-newCol') ;...
        shur*(-brep+newCol')    ];
    
    %%  Update record to include what we just added
    C = [C newCol];
    indices = [indices newColIndex];
    
    %%  Select new column to be used on next iteration
    delta = sum(R.*C',1)-D;
    
    %%  Grab out the row with max norm
    [m,newColIndex] = max(abs(delta));
    if verbose;
        fprintf('.');
        if mod(size(R,1),50)==1
            fprintf('\n\t\t\t');
        end
    end
    
end
indices = indices(1:L);
if verbose; fprintf('\n\tEnd oASIS: time = %f secs\n',toc); end
return



%%  Randomly sample columns, from precomputed X or from data and a kernel function
function [selection, C]= randomSample(X,L, verbose, use_randomseed)
if use_randomseed; rng(0,'twister'); end

if verbose; fprintf('\tBegin Random...\n'); tic; end

if isnumeric(X)
    N = size(X,2);
    selection = randperm(N);
    selection = selection(1:L);
    C = X(:,selection);
else
    if verbose; fprintf('\t\tSelecting Cols'); end
    N = length(X('D','D'));
    selection = randperm(N);
    selection = selection(1:L);
    C = zeros(N,L);
    for i=1:L
        C(:,i) = X([],selection(i));
        if verbose
            if mod(i,50)==1
                fprintf('\n\t\t\t');
            end
            fprintf('.');
        end
    end
    fprintf('\n');
end

if verbose; fprintf('\tEnd Random: time = %f secs\n',toc); end

return



%% Fill out the options with defaults
function opts = fillDefaultOptions(opts)

% Display info?
if ~isfield(opts,'verbose')
    opts.verbose = false;
end

% Expand the Nystrom approximation to Gtilde = C*W^{-1}C?
if ~isfield(opts,'expandNystrom')
    opts.expandNystrom = false;
end

%  Compute the approximate singular vals/vectors
if ~isfield(opts,'computeApproxSVD')
    opts.computeApproxSVD = false;
end

% columns to sample - if the user wants to hand this in explicitly
if ~isfield(opts,'selection')
    opts.selection = [];
end

% Use the MEX interface for oASIS, or use the MATLAB interface
if ~isfield(opts,'use_mex')
    opts.use_mex = true;
end

% For repeatable results
if ~isfield(opts,'use_randomseed')
    opts.use_randomseed = true;
end


% For repeatable results
if ~isfield(opts,'startSize')
    opts.startSize = 1;
end

return




