%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FCSC hybrid ICA procedure adapted from Amico & Goñi, Network Neuroscience 2018 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% batch example of the "hybrid "connICA (Amico & Goñi, Network Neuroscience 2018)
    %% Enrico Amico & Joaquín Goñi, Purdue University
    %% Thanks to Jonathan Wirsich (Beckman Institute, UIUC) for insightful comments and discussions
    %% version 1.0. April 10, 2018
    %% IMPORTANT: FastICA Matlab package IS NEEDED!
    % Please download FastICA package
    % https://research.ics.aalto.fi/ica/fastica/code/dlcode.shtml

    %% initialize environment
    close all;
    clearvars
    clc;
    
    %% changing default fontsize
    fontsize = 18;
    set(0,'DefaultAxesFontName','Times New Roman');
    set(0,'DefaultAxesFontSize',fontsize-2);
    
    set(0,'DefaultTextFontname','Times New Roman');
    set(0,'DefaultTextFontSize',fontsize);

    %% Load FC data (functional connectivity) and SC data (structural connectivity).
    %  dimensions are brain regions x brain regions x subjects
    FC = [];
    SC = [];
    FC = load('FC_99.mat');
    SC = load('SC_99.mat');
    FC = FC.A1;
    SC = SC.A1;
    
    %% Configuration
    addpath(genpath('FastICA_25')); %add FastICA folder to path.
    % ConnICA params
    configs.numRegions = 268; % Shen268 parcellation
    configs.mask_ut = triu(true(configs.numRegions,configs.numRegions),1); % upper triangular mask, for FC data
    configs.mask_lt = tril(true(configs.numRegions,configs.numRegions),-1); % lower triangular mask, for SC data
    
    configs.numConn = size(FC,3); % number of connectomes (subjects or sessions)

    % Compute group average SC mask of connected edges (see for details Amico & Goni, Network Neuroscience 2018)
    mask_nonzeroSC = true(configs.numRegions,configs.numRegions);
    for i=1:configs.numConn
         SC_i = SC(:,:,i);
         mask_nonzeroSC = mask_nonzeroSC & (SC_i>0);
    end
    
    %% create connICA hybrid input matrix (nSubj x (2 x n FC-SC edges))
    connICA_hybrid = zeros(configs.numConn,1*nnz(configs.mask_ut));
    disp('Creating hybrid connICA matrix...')
    for i=1:configs.numConn
         
        aux = SC(:,:,i);
        %% Compute SC "correlation node to node" (see Amico & Goñi, Network Neuroscience 2018)
        % Note: other data normalization can be used here
        SC_corr = f_compute_SC_corr(aux);
        SC_part = SC_corr(configs.mask_lt);
        % concatenate FC and SC                
        connICA_hybrid(i,:) = SC_part';
    end
    disp('Done.');
    %% For the SC part, consider only structurally connected edges when extracting hybrid traits
    %% (see for details Amico & Goñi, Network Neuroscience 2018)
     connICA_hybrid(:,[ ~mask_nonzeroSC(configs.mask_lt)'])= []; 
     
    %% FastICA params
    configs.epsilon = 0.0001;
    configs.maxNumIterations = 1000;
    configs.maxFinetune = 1000;
    configs.numRuns = 1000; %100; %run FastICA several time for robust comps extraction  
    configs.numOfIC = 25; % number independent components
    
    flags.PCA = 1; % if 1, perform PCA before ICA
    configs.PCAthr = 0.90; % Explained variance threshold for PCA. Feel free to explore 0.75 or .80
    % perform PCA before ICA
    if flags.PCA==1
        disp('running PCA compression before ICA...');
        numFCs = size(connICA_hybrid,1);
        numPCAComps = size(connICA_hybrid,1);
        [~, ~, latent] = pca(connICA_hybrid','NumComponents',numPCAComps); 
        variance = cumsum(latent)./sum(latent); 
        variance = variance(1:numPCAComps); % explained variance with the selected num of PCA comps
        numPCAComps = find(variance>=configs.PCAthr,1);
        disp('# PCA comps retained:');
        disp(numPCAComps);
        if numPCAComps<configs.numOfIC
            error('Error: PCA comps retained lower than number of requested ICA comps! Check the configs.numOfIC and try again');
        end
        if numPCAComps==configs.numOfIC
            warning(sprintf('Warning: number of PCA comps retained equal to the number of requested ICA comps.\n Please note that, in our experience, sometimes FastICA does not reach convergence under this configuration')); %#ok<SPWRN>
        end
        
        %figure, plot(1:numFCs,variance,'ok'); % uncomment this line if you want to see the variance explained plot
        [COEFFS, SCORE, latent] = pca(connICA_hybrid','NumComponents',numPCAComps);     
        PCA_clean_matrix = SCORE * COEFFS'; % PCA reconstructed demeaned data
        PCA_clean_matrix = bsxfun(@plus, PCA_clean_matrix,mean(connICA_hybrid,2)'); % plug the mean back
        PCA_clean_matrix = PCA_clean_matrix'; % back to subjects x edges form
        connICA_hybrid = PCA_clean_matrix;
        disp('Done.');
    end
    
    
    icasig = nan(size(connICA_hybrid,2),configs.numOfIC,configs.numRuns);
    A = nan(size(connICA_hybrid,1),configs.numOfIC,configs.numRuns);
    for i=1:configs.numRuns
        [icasig_onerun,A_onerun,~] = fastica(connICA_hybrid,'approach','symm','numOfIC',configs.numOfIC,'verbose','off',...
            'epsilon',configs.epsilon,'maxNumIterations',configs.maxNumIterations,...
            'maxFinetune',configs.maxFinetune);%running fastica
        A(:,:,i) = single(A_onerun);% connICA weights
        icasig(:,:,i) = single(icasig_onerun'); % connICA single run traits
        if mod(i,25)==0
            disp(sprintf('%d runs out of %d',i,configs.numRuns)); %#ok<DSPS>
        end
    
    end
    %% robust traits extraction criteria (see Amico et al., NeuroImage 2017 and Amico & Goñi, Network Neuroscience 2018)
    configs.minFreq = 0.95; % minimum frequency required 
    configs.corrMin = 0.95; % correlation between FC_traits
    configs.corrMin_A = 0.95; % correlation between subject weights
    [comp_match_run1, freq] = run_robust_connICA_fast(A,icasig,configs); % comp_match_run1 stores the robust component ID per connICA run
    aux=(freq>configs.minFreq);
    RC_Index=find(aux(:,1)); % RC_Index == Robust Components Index
    disp(RC_Index)
    if isempty(RC_Index)
        error('No Robust FC traits found!');
    end
    %% Put back the robust traits in matrix form
    RC = struct;
    for t=1:length(RC_Index)
        compIndex=RC_Index(t); % choose the component(order of component) we want to look at 
        figure,
        icasig_comp = nan(configs.numRuns,size(connICA_hybrid,2));
        a0 = A(:,comp_match_run1(compIndex,1),1); %this is used as reference so that the same component across runs is always positively correlated (not inverted)
        weights = nan(size(A,1),configs.numRuns);
        for i=1:configs.numRuns
            if comp_match_run1(compIndex,i)>0
                a = A(:,comp_match_run1(compIndex,i),i);
                icasig_comp_one = squeeze(icasig(:,comp_match_run1(compIndex,i),i));
                if corr(a,a0)>0
                    plot(a); hold on;
                    icasig_comp(i,:) = icasig_comp_one;
                else
                    plot(-a); hold on;
                    a= -a;
                    icasig_comp(i,:) = -icasig_comp_one;
                end
                weights(:,i) = a;
            end
        end
        ylabel('weights');
        xlabel('subjects');
        title(sprintf('connICA comp %d',compIndex));
        %% Plot hybrid traits
        RC(t).hybrid_vector = nanmean(icasig_comp); % avg per column (across runs) 
        RC(t).FC_vector = [];
        RC(t).FC_vector = RC(t).hybrid_vector(1:nnz(configs.mask_ut));
        RC(t).FC_matrix = zeros(configs.numRegions,configs.numRegions);
        RC(t).FC_matrix(configs.mask_ut)= RC(t).FC_vector; %fill upper triangular (FC part)
        RC(t).FC_matrix = RC(t).FC_matrix + (RC(t).FC_matrix'); % symmetrize matrix
        RC(t).SC_vector = [];
        RC(t).SC_vector =  RC(t).hybrid_vector(1:27236);
        RC(t).SC_matrix = zeros(configs.numRegions,configs.numRegions);
        RC(t).SC_matrix(configs.mask_lt & mask_nonzeroSC)= RC(t).SC_vector; % fill lower triangular (SC part)
        RC(t).SC_matrix = RC(t).SC_matrix + (RC(t).SC_matrix'); % symmetrize matrix    
        RC(t).weights = weights; % Hybrid trait vector of weights, per run    
        
    end
    %% 
    
    for t=1:length(RC)
    
    
        figure; 
         subplot(1,2,1); imagesc(RC(t).FC_matrix,[-3,3]); colormap jet; colorbar; axis square;
         set(gca,'xtick',[]); set(gca,'ytick',[]); xlabel('regions'); ylabel('regions');
         title('FC part')
        subplot(1,2,2); imagesc(RC(t).SC_matrix,[-3,3]); colormap jet; colorbar; axis square;
        set(gca,'xtick',[]); set(gca,'ytick',[]); xlabel('regions'); ylabel('regions');
        title('SC part')
        %subtitle(sprintf('connICA hybrid trait %d',compIndex));
    end
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CCA procedure adapted from Smith et al., 2015 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% additional matlab toolboxes required (will need to addpath for each of these)
% FSLNets     http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLNets
% PALM        http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM
% nearestSPD  http://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
                                             
%% ICA component derived from average weight of coomponents across 1000 runs for each subject
Nica=4;

ERmat = zeros(99, Nica);
for q = 1:99
    disp(q)
    for u = 1:Nica
        ERmat(:, u) = mean(RC(u).weights, 2, 'omitnan');
    end
end

%% setup confounds matrix and regress out from behavioral and neural data
conf=palm_inormal([ covar ]);    % Gaussianise
ERb=ERpca3-conf*(pinv(conf)*ERpca3)
ERm=ERmat-conf*(pinv(conf)*ERmat)

%% CCA
[grotA,grotB,grotR,grotU,grotV,grotstats]=canoncorr(ERm,ERb);

Nperm=10000;  
PAPset=palm_quickperms(99, ones(99,1), Nperm); 

%%% CCA permutation testing
Nkeep=3;
                                             
grotRp=zeros(Nperm,Nkeep); clear grotRpval;
for j=1:Nperm
  j
  [grotAr,grotBr,grotRp(j,:),grotUr,grotVr,grotstatsr]=canoncorr(ERm,ERb(PAPset(:,j)));
end
for i=1:Nkeep;  % get FWE-corrected pvalues
  grotRpval(i)=(1+sum(grotRp(2:end,1)>=grotR(i)))/Nperm;
end
grotRpval
Ncca=sum(grotRpval<0.05)  % number of FWE-significant CCA components

%% results
clearvars grotAA grotBB

for t=1:Ncca
    %%% netmat weights for CCA mode 1
    grotAA(:,:,t) = corr(grotU(:,t),ERm)';

    %%% SM weights for CCA mode 1
    grotBB(:,:,t) = corr(grotV(:,t),palm_inormal(ERb),'rows','pairwise');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calculating network score for HBN subjects %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% finding the composite FC network weights
combFC = zeros(268,268);
H1_FC=readmatrix('H1_FC.csv');
H2_FC=readmatrix('H2_FC.csv');
H3_FC=readmatrix('H3_FC.csv');
H4_FC=readmatrix('H4_FC.csv');
combFC = 0.2753*H1_FC + 0.3588*H2_FC + (-0.9404)*H3_FC + (-0.0250)*H4_FC;

%% multiplying composite network weights to HBN connectomes
B1=A1;
for i=1:93
    
    B1(:,:,i)=A1(:,:,i).*combFC;

end

%% sum of absolute values
pos=ones(93);
neg=ones(93);
all=ones(93);
for i=1:93
    temp = B1(:,:,i);
    pos(i) = sum(temp(temp >= 0));
    neg(i) = sum(temp(temp < 0));
    all(i) = p(i)-n(i);
end

%% writematrix
writematrix(pos(:,1), "HBN_psum.csv")
writematrix(neg(:,1), "HBN_nsum.csv")
writematrix(all(:,1), "HBN_sum.csv")

