% oASIS: Adaptive Column Sampling for Kernel Matrix Approximation
% R. Patel, T. Goldstein, E. Dyer, A. Mirhoseini, and R. Baraniuk
% Submitted to IEEE JSTSP
%
% Demo for explicit kernel matrices. See paper for details.
%

clear
close all
clc

addpath('supportscripts');



%% Set data options 
%  A dataset of 10 points per vertex of an 8-dimensional hypercube.
datasetName = 'BORG';
numdimensions = 8;
pts_per_dimension = 10;
sigfactor = 8;
matrixType = 'gaussian';%{'gaussian', 'diffusion', 'gram'};



%% Set approximation options. See "help oASIS" for details.
Lvec = [50:50:450];

opts = [];
opts.verbose = true;            
opts.expandNystrom = false;    
opts.computeApproxSVD = false;
opts.selection = [];           
opts.use_mex = false;           
opts.use_randomseed = true;
opts.startSize = 1;  



%% Create Full, Precomputed kernel matrix
fprintf('Loading Data Set %s\n', datasetName);
Z = assimilate_BORG(numdimensions,pts_per_dimension);

switch matrixType
    
    case 'gaussian'
        D = squareform(pdist(Z'));  
        fprintf('Computing Sigma...');
        sig = max(D(:))/sigfactor;
        fprintf('%d\n',sig);
        S = exp( (-D.^2)/2/sig^2 ); 
        clear D
        
        G = S;
        
    case 'diffusion'
        D = squareform(pdist(Z'));  
        fprintf('Computing Sigma...');
        sig = max(D(:))/sigfactor;
        fprintf('%d\n',sig);
        S = exp( (-D.^2)/2/sig^2 );
        clear D
        
        fprintf('Normalizing...\n');
        Nmlz = diag( sum(S).^-.5 );
        G = Nmlz*S*Nmlz;  
        
    case 'gram'
        G = Z'*Z;
        
    otherwise
        assert(0,['Invalid matrix type: ' matrixType]);
end

nmG = norm(G,'fro');
clear S

col_selection = zeros(2,Lvec(end));
error = zeros(length(Lvec),2);



%% Run oASIS and Uniform Random Sampling, and compute errors
[outs_oasis] = oASIS( G, Lvec, 'oASIS', opts);

if length(Lvec) == 1
    Gtilde = outs_oasis.C*(outs_oasis.W\outs_oasis.C');
    error(1) = norm(Gtilde-G,'fro')/nmG; %Compute full relative error as we can compute all of G
    fprintf('\t\t%d: error = %d\n',Lvec,error(1));
else
    % If we chose more than one sample size, we want to loop through all of
    % them
    for trialNum = 1:length(Lvec)
        nys = outs_oasis.nystroms{trialNum};
        Gtilde = nys.C*(nys.W\nys.C');
        error(trialNum,1) = norm(Gtilde-G,'fro')/nmG;
        fprintf('\t\t%d: error = %d\n',nys.L,error(trialNum,1));
    end
end
col_selection(1,:) = outs_oasis.selection;


[outs_random] = oASIS( G, Lvec, 'random', opts);

if length(Lvec) == 1
    Gtilde = outs_random.C*(outs_random.W\outs_random.C');
    error(2) = norm(Gtilde-G,'fro')/nmG; %Compute full relative error as we can compute all of G
    fprintf('\t\t%d: error = %d\n',Lvec,error(2));
else
    % If we chose more than one sample size, we want to loop through all of
    % them    
    for trialNum = 1:length(Lvec)
        nys = outs_random.nystroms{trialNum};
        Gtilde = nys.C*(nys.W\nys.C');
        error(trialNum,2) = norm(Gtilde-G,'fro')/nmG;
        fprintf('\t\t%d: error = %d\n',nys.L,error(trialNum,2));
    end
    
end
col_selection(2,:) = outs_random.selection;



%% Plot errors
if length(Lvec) > 1
    
    markers = {'ko-','+-'};
    colors = [0 0 0; 0 0 1];
    fsz = 20;
    
    iinds = [1 2];
    
    figure(1);clf
    set(gcf,'Position',[ 7  212 579 286]);
    clf;hold off;
    for errind = iinds;
        semilogy(Lvec,error(:,errind),markers{iinds(errind)}, ...
            'Color',colors(iinds(errind),:), ...
            'LineWidth',1.5,'MarkerSize',14);
        hold all
    end
    grid on
    set(gca,'FontSize',fsz)
    
    [~,~,~,~] = legend({'oASIS','random'}, ...
        'Interpreter','Latex','Location', 'SW','fontsize',fsz-4);
    xlim([0 Lvec(end)]);
    
    xlabel('number of samples $\ell$','Interpreter','Latex','fontsize',fsz);
    ylabel('relative error','Interpreter','Latex','fontsize',fsz);
    title([datasetName ' ' matrixType],'Interpreter','Latex','fontsize',fsz);
    pbaspect([2 1 1]);
    drawnow;
end


clear colors errind fsz iinds markers trialNum nys