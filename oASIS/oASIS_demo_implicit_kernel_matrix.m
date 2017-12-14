% oASIS: Adaptive Column Sampling for Kernel Matrix Approximation
% R. Patel, T. Goldstein, E. Dyer, A. Mirhoseini, and R. Baraniuk
% Submitted to IEEE JSTSP
%
% Demo for implicit kernel matrices. See paper for details.
%

clear
close all
clc

addpath('supportscripts');



%% Set data options
%  A dataset of 30 points per vertex of an 8-dimensional hypercube.
datasetName = 'BORG';
numdimensions = 8;
pts_per_dimension = 30;
sigfactor = 8;
matrixType = 'gaussian';



%% Set approximation options. See "help oASIS" for details.
Lvec = [50:50:450];

opts = [];
opts.verbose = true;            
opts.computeApproxSVD = false;  
opts.selection = [];            
opts.use_randomseed = true;    
opts.startSize = 1;          



%% Create/Load Data
fprintf('Loading Data Set %s\n', datasetName);
Z = assimilate_BORG(numdimensions,pts_per_dimension);

%% Because computing the entire Sigma is difficult we use a subset
fprintf('Computing Sigma...');
rng(0,'twister');
subset = Z(:,randi(size(Z,2),5000,1))';
D = squareform(pdist(subset));
sig = max(D(:))/sigfactor;
fprintf('%d\n',sig);
clear subset D


%% Kernel function is embedded in a function handle with data Z
switch matrixType
    case 'gaussian'
        G = @(r,c) GaussianKernelMatrixSampler(Z,r,c,sig);
    otherwise
        assert(0,['Invalid matrix type: ' matrixType]);
end

col_selection = zeros(2,Lvec(end));
error = zeros(length(Lvec),2);



%% Run oASIS and Uniform Random Sampling, and compute error
[outs_oasis] = oASIS( G, Lvec, 'oASIS', opts);

fprintf('\tCalculating Errors\n');
if length(Lvec) == 1
    [relApproxError] = calcApproximationError(G,outs_oasis);
    error(1) = relApproxError;
    fprintf('\t\t%d: error = %d \n',Lvec,relApproxError);
else
    for trialNum = 1:length(Lvec)
        nys = outs_oasis.nystroms{trialNum}; %for more than one sample size, nystroms doesn't store W+ so we have to do it outside
        [Uw,Sw,Vw] = svd(nys.W);
        Splus = zeros(size(Sw));
        Splus(Sw>1e-7) = Sw(Sw>1e-7).^-1;
        nys.Wplus = Uw*Splus*Vw';
        clear Uw Sw Vw Splus
        
        %Calculate relative error over 100,000 points
        [relApproxError] = calcApproximationError(G,nys);
        error(trialNum,1) = relApproxError;
        
        fprintf('\t\t%d: error = %d \n',nys.L,relApproxError);
    end
end
col_selection(1,:) = outs_oasis.selection;


%Now sample and compute error for uniform random sampling
[outs_random] = oASIS( G, Lvec, 'random', opts);

fprintf('\tCalculating Errors\n');
if length(Lvec) == 1
    [relApproxError] = calcApproximationError(G,outs_random);
    error(2) = relApproxError;
    fprintf('\t\t%d: error = %d \n',Lvec,relApproxError);
else
    for trialNum = 1:length(Lvec)
        nys = outs_random.nystroms{trialNum};
        [Uw,Sw,Vw] = svd(nys.W);
        Splus = zeros(size(Sw));
        Splus(Sw>1e-7) = Sw(Sw>1e-7).^-1;
        nys.Wplus = Uw*Splus*Vw';
        clear Uw Sw Vw Splus
        
        %Calculate relative error over 100,000 points
        [relApproxError] = calcApproximationError(G,nys);
        error(trialNum,2) = relApproxError;
        
        fprintf('\t\t%d: error = %d \n',nys.L,relApproxError);
    end
end
col_selection(2,:) = outs_random.selection;



%% Plot errors
if length(Lvec)>1
    
    markers = {'ko-','+-'};
    colors = [0 0 0; 0 0 1];
    fsz = 20;
    
    iinds = [1 2];
    
    figure(1);clf
    set(gcf,'Position',[ 7  212 579 286]);
    clf;hold off;
    for errind = iinds;
        semilogy(Lvec,error(:,errind),markers{iinds(errind)}, ...
            'Color',colors(iinds(errind),:),'LineWidth',1.5,'MarkerSize',14);
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