% Author : Muhammad Ahmad
% Date   : 25/07/2020
% Email  : mahmad00@gmail.com
% Reference: Spatial-prior generalized fuzziness extreme learning machine...
% autoencoder-based active learning for hyperspectral image classification
% https://www.sciencedirect.com/science/article/abs/pii/S0030402619316109
% https://www.researchgate.net/publication/338641736_Spatial-prior_...
% Generalized_Fuzziness_Extreme_Learning_Machine_Autoencoder-based_Active...
% _Learning_for_Hyperspectral_Image_Classification
%% Clear and Cloase already opened Figures
clc; clear; close all;
%% Warning off
warning('off', 'all');
%% Loading Data along with required Parameters
load('Data');
%% Active Learning Classifier
[Accuracy, Time] = MY_ML_ELM_AL(img, TrC, TeC, AL_Strtucture, Samples, ...
    Fuzziness, Parameters, folder, gt);
%% plot the Results
figure(1)
set(gcf,'color','w');
set(gca, 'fontsize', 12, 'fontweight','bold')
hold on
plot(Accuracy(:,1), '--s','MarkerSize', 8,...
    'MarkerEdgeColor','red', 'LineWidth', 2.5)
hold on
plot(Accuracy(:,2), '-.o','MarkerSize', 8,...
	'MarkerEdgeColor','blue', 'LineWidth', 2.5)
hold on
plot(Accuracy(:,3), ':*','MarkerSize', 8,...
	'MarkerEdgeColor','black', 'LineWidth', 2.5)
hold on
plot(Accuracy(:,4), '-.+','MarkerSize', 8,...
	'MarkerEdgeColor','cyan', 'LineWidth', 2.5)
hold on
plot(Accuracy(:,5), '-d','MarkerSize', 8,...
        'MarkerEdgeColor','green', 'LineWidth', 2.5)
hold on
plot(Accuracy(:,6), '-<','MarkerSize', 8,...
        'MarkerEdgeColor','black', 'LineWidth', 2.5)
hold on
legend({'OA Train', 'OA Test', 'AA Train', 'AA Test',...
    'kappa Train', 'kappa Test'},'FontSize',12,...
        'FontWeight','bold','Location','southeast', 'color','k');
legend('boxoff'); grid on;
ylabel('Accuracy','FontSize',12,'FontWeight','bold', 'color','k')
xlabel('Number of Iterations','FontSize',12,'FontWeight','bold', 'color','k')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(figure(1), 'Accuracy.png');
pause(1)
close all;
%% Training and Test Time
figure(1)
set(gcf,'color','w');
set(gca, 'fontsize', 12, 'fontweight','bold')
hold on
plot(Time(:,1), '--s','MarkerSize', 8,...
    'MarkerEdgeColor','red', 'LineWidth', 2.5)
hold on
plot(Time(:,2), '-.o','MarkerSize', 8,...
	'MarkerEdgeColor','blue', 'LineWidth', 2.5)
hold on
legend({'Train', 'Test'},'FontSize',12,...
	'FontWeight','bold','Location','southeast', 'color','k');
legend('boxoff'); grid on;
ylabel('Time in Seconds','FontSize',12,'FontWeight','bold', 'color','k')
xlabel('Number of Iterations','FontSize',12,'FontWeight','bold', 'color','k')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(figure(1), 'Time.png');
pause(1)
close all; clear;
%% Internal Functions
function [Acc, Time] = MY_ML_ELM_AL(varargin)
%% Compiling the Input
img = varargin{1};
if ~numel(img)
    error('Please Provide HSI Data');
end
%% 
TrC = varargin{2};
if isempty(TrC)
    error('Please Provide Training Labels');
end
%%
TeC = varargin{3};
if isempty(TeC)
    error('Please Provide Test Labels');
end
%% 
AL_Strtucture = varargin{4};
if isempty(AL_Strtucture)
    error('Please Provide Active Learning Parameters');
else
    tot_sim = AL_Strtucture.M/AL_Strtucture.h + 1;
end
%%
Samples = varargin{5};
if isempty(Samples)
    error('Please provide the Sampling Technique');
end
%%
Fuzziness = varargin{6};
if isempty(Fuzziness)
    error('Please Provide the Fuzziness Catagorization');
end
%%
Parameters = varargin{7};
if isempty(Parameters)
    error('Plesae Provide the ELM Parameters');
end
%%
folder = varargin{8};
if isempty(folder)
    error('Please Provide the Directory to Save the Results');
end
%%
gt = varargin{9};
if isempty(gt)
    error('Please Provide the Ground Truths');
end
%% Saving Test Samples Locations
TeC_Locations = cell(tot_sim, 1);
ELM_Per_Clas = cell(tot_sim, 1);
ELM_Tr_Per_Clas = cell(tot_sim, 1);
%% Start Active Learning Process Multi Layered ELM
for iter = 1 : tot_sim
    fprintf('ELM Active Selection %d \n', iter)
    Tr = img(TrC(1,:), :);
    Te = img(TeC(1,:), :);
    TeC_Locations{iter} = TeC;
%% Multi Layer ELM
    [ELM__W_Tr, ELM_W, TrT, TeT, ~,~] = ELM_AE(Tr', TrC(2, :),...
        Te', TeC(2,:), Parameters.TLs, Parameters.HNs,...
            Parameters.Regu, Parameters.Rho, Parameters.Sigpara, ...
                Parameters.sigpara1, Parameters.AF);
    ELM_Class_Results.Time(iter,:) = [TrT TeT];
%% Compute the Output Class
    [~, ELM_Class_Results.map] = max(ELM_W);
    ELM_Per_Clas{iter} = ELM_Class_Results.map;
%% Compute the Accuracy
    uc = unique(TrC(2, :));
    [ELM_Class_Results.OA(iter), ELM_Class_Results.kappa(iter),...
        ELM_Class_Results.AA(iter), ELM_Class_Results.CA(iter,:)] = ...
            My_Accuracy(TeC(2,:)-1, ELM_Class_Results.map-1,(1:numel(uc)));
    %% Training Accuracy
    [~, ELM_Tr_Class_Results.map] = max(ELM__W_Tr);
    ELM_Tr_Per_Clas{iter} = ELM_Tr_Class_Results.map;
    %% Training Accuracy
    [ELM_Tr_Class_Results.OA(iter), ELM_Tr_Class_Results.kappa(iter),...
        ELM_Tr_Class_Results.AA(iter), ELM_Tr_Class_Results.CA(iter,:)] = ...
            My_Accuracy(TrC(2,:)-1, ELM_Tr_Class_Results.map-1,(1:numel(uc)));        
    %% Active Learning Sampels Selection    
            ELM_W = My_Member(uc, ELM_W');
            ELM_Fuz = My_Fuzziness(ELM_W);
            Pred = ELM_Class_Results.map;
            Pred  = [Pred; AL_Strtucture.Candidate_Set];
            Pred = [ELM_Fuz'; Pred]';
            [A, ind] = sortrows(Pred, -1); %% High Fuzziness Samples
            [idx, ~] = find(A(:,4) ~= A(:,2)); %% Mis-Classified Samples
            index_ELM_minME = ind(idx);
            if length(index_ELM_minME)>(AL_Strtucture.h)
                xp = index_ELM_minME(1 : AL_Strtucture.h)';
            else
                ind(idx) = [];
                index_ELM_minME = [index_ELM_minME' ind'];
                xp = index_ELM_minME(1 : AL_Strtucture.h)';
            end
            TrCNew = AL_Strtucture.Candidate_Set(:,xp);
            TrC = [TrC, TrCNew];
            AL_Strtucture.Candidate_Set(:,xp) = [];
            TeC = AL_Strtucture.Candidate_Set;
%% End For Iterations on AL. 
end
%% Training
Tr_OA = ELM_Tr_Class_Results.OA';
Tr_AA = ELM_Tr_Class_Results.AA';
Tr_kappa = ELM_Tr_Class_Results.kappa';
%% Test
Te_OA = ELM_Class_Results.OA';
Te_AA = ELM_Class_Results.AA';
Te_kappa = ELM_Class_Results.kappa';
%% Combining
Acc = [Tr_OA Te_OA Tr_AA Te_AA Tr_kappa Te_kappa];
Time = ELM_Class_Results.Time;
end
%% MLELM-AE Function
function [Y, TY, TrainingTime, TestingTime, TrainingAccuracy, ...
            TestingAccuracy] = ELM_AE(Tr, TrC, Te, TeC, TotalLayers, ...
                HiddernNeurons, C1, rhoValue, sigpara, sigpara1,...
                    ActivationFunction)
P = Tr;
T = TrC;
TV.T = TeC;
TV.P = Te;
NumberofTrainingData = size(P,2);
NumberofTestingData = size(TV.T,2);
NumberofInputNeurons = size(P,1);
clear Tr TrC Te TeC
%% Preprocessing the data of classification
label = unique(T);
label1 = unique(TV.T);
if label ~= label1
    error('Error. Number of Training and Test Classes Must be Same')
end
clear label1
number_class = numel(label);
NumberofOutputNeurons = number_class;
%% Processing the targets of training
temp_T = zeros(NumberofOutputNeurons, NumberofTrainingData);
for i = 1:NumberofTrainingData
    for j = 1:number_class
        if label(1,j) == T(1,i)
            break; 
        end
    end
    temp_T(j,i) = 1;
end
T = temp_T*2-1;
%% Processing the targets of testing
temp_TV_T = zeros(NumberofOutputNeurons, NumberofTestingData);
for i = 1:NumberofTestingData
    for j = 1:number_class
        if label(1,j) == TV.T(1,i)
            break; 
        end
    end
    temp_TV_T(j,i) = 1;
end
TV.T = temp_TV_T*2-1;
%% Calculate weights & biases
train_time = tic;
no_Layers = TotalLayers;
stack = cell(no_Layers+1,1);    
lenHN = length(HiddernNeurons);
lenC1 = length(C1);
lensig = length(sigpara);
lensig1 = length(sigpara1);
HN_temp = [NumberofInputNeurons,HiddernNeurons(1:lenHN-1)];
if length(HN_temp) < no_Layers
    HN = [ HN_temp, repmat( HN_temp( length( HN_temp ) ),1,no_Layers-length(HN_temp) ), HiddernNeurons(lenHN) ];
    C = [C1(1:lenC1-2), zeros(1,no_Layers - length(HN_temp)  ), C1(lenC1-1:lenC1) ];
    sigscale = [sigpara(1:lensig-1),ones(1,no_Layers - length(HN_temp)),sigpara(lensig)];
    sigscale1 = [sigpara1(1:lensig1-1),ones(1,no_Layers - length(HN_temp)),sigpara1(lensig1)];
else
    HN = [NumberofInputNeurons,HiddernNeurons];
    C = C1;
    sigscale = sigpara;
    sigscale1 = sigpara1;
end
clear HN_temp;
InputDataLayer = zscore(P);
clear P; 
rng('default');
for i = 1:1:no_Layers
    InputWeight = rand(HN(i+1),HN(i))*2 -1;
    if HN(i+1) > HN(i)
        InputWeight = orth(InputWeight);
    else
        InputWeight = orth(InputWeight')';
    end
    BiasofHiddenNeurons = rand(HN(i+1),1)*2 -1;
    BiasofHiddenNeurons = orth(BiasofHiddenNeurons);
    tempH = InputWeight*InputDataLayer; 
    clear InputWeight;
    ind = ones(1,NumberofTrainingData);
    BiasMatrix = BiasofHiddenNeurons(:,ind);
    tempH = tempH+BiasMatrix;
    clear BiasMatrix BiasofHiddenNeurons
    fprintf(1,'AutoEncorder Max Val %f Min Val %f\n',max(tempH(:)),min(tempH(:)));
    %% Calculate hidden neuron output matrix H
    switch lower(ActivationFunction)
        case {'sig','sigmoid'}
            %%%%%%%% Sigmoid
            H = 1 ./ (1 + exp(-sigscale1(i)*tempH));
        case {'sin','sine'}
            %%%%%%%% Sine
            H = sin(sigscale1(i)*tempH);
        case {'hardlim'}
            %%%%%%%% Hard Limit
            H = double(hardlim(sigscale1(i)*tempH));
        case {'tribas'}
            %%%%%%%% Triangular basis function
            H = tribas(sigscale1(i)*tempH);
        case {'radbas'}
            %%%%%%%% Radial basis function
            H = radbas(sigscale1(i)*tempH);
    end
    clear tempH;
%% Calculate output weights OutputWeight (beta_i)
    if HN(i+1) == HN(i)
        [~,stack{i}.w,~] = procrustNew( InputDataLayer',H');
    else
        if C(i) == 0
            stack{i}.w =pinv(H') * InputDataLayer';
        else
            rhohats = mean(H,2);
            rho = rhoValue;
            KLsum = sum(rho * log(rho ./ rhohats) + (1-rho) * log((1-rho) ./ (1-rhohats)));
            Hsquare =  H * H';
            HsquareL = diag(max(Hsquare,[],2));
            stack{i}.w = ((eye(size(H,1)).*KLsum +HsquareL)*(1/C(i))+Hsquare) \ (H * InputDataLayer');
            clear Hsquare HsquareL;
        end
    end
    tempH =(stack{i}.w) *(InputDataLayer);
    clear InputDataLayer;
    if HN(i+1) == HN(i)
        InputDataLayer = tempH;
    else
        fprintf(1,'Layered Max Val %f Min Val %f\n',max(tempH(:)),min(tempH(:)));
    %% Calculate hidden neuron output matrix H
    switch lower(ActivationFunction)
        case {'sig','sigmoid'}
            %%%%%%%% Sigmoid
            InputDataLayer = 1 ./ (1 + exp(-sigscale(i)*tempH));
        case {'sin','sine'}
            %%%%%%%% Sine
            InputDataLayer = sin(sigscale(i)*tempH);
        case {'hardlim'}
            %%%%%%%% Hard Limit
            InputDataLayer = double(hardlim(sigscale(i)*tempH));
        case {'tribas'}
            %%%%%%%% Triangular basis function
            InputDataLayer = tribas(sigscale(i)*tempH);
        case {'radbas'}
            %%%%%%%% Radial basis function
            InputDataLayer = radbas(sigscale(i)*tempH);
    end   
% InputDataLayer =  1 ./ (1 + exp(-sigscale(i)*tempH));
    end
    clear tempH H;
end
if C(no_Layers+1) == 0
    stack{no_Layers+1}.w = pinv(InputDataLayer') * T';
else
    stack{no_Layers+1}.w = (eye(size(InputDataLayer,1))/C(no_Layers+1)+InputDataLayer * InputDataLayer') \ (InputDataLayer * T');  
end
TrainingTime = toc(train_time);
%% Display the Network
% display_network(orth(stack{1}.w'));
%% Calculate the training accuracy
Y = (InputDataLayer' * stack{no_Layers+1}.w)';
clear InputDataLayer H;
%% Calculate the output of testing input
test_time = tic;
InputDataLayer = zscore(TV.P);
clear TV.P;
for i = 1:1:no_Layers   
    tempH_test = (stack{i}.w)*(InputDataLayer);
    clear TV.P;
    if HN(i+1) == HN(i)
        InputDataLayer = tempH_test;
    else
        %% Calculate hidden neuron output matrix H
        switch lower(ActivationFunction)
            case {'sig','sigmoid'}
                %%%%%%%% Sigmoid
                InputDataLayer = 1 ./ (1 + exp(-sigscale(i)*tempH_test));
            case {'sin','sine'}
                %%%%%%%% Sine
                InputDataLayer = sin(sigscale(i)*tempH_test);
            case {'hardlim'}
                %%%%%%%% Hard Limit
                InputDataLayer = double(hardlim(sigscale(i)*tempH_test));
            case {'tribas'}
                %%%%%%%% Triangular basis function
                InputDataLayer = tribas(sigscale(i)*tempH_test);
            case {'radbas'}
                %%%%%%%% Radial basis function
                InputDataLayer = radbas(sigscale(i)*tempH_test);
        end
    end
    clear tempH_test;
end
TY = (InputDataLayer' * stack{no_Layers+1}.w)';
TestingTime = toc(test_time);
%% Training Accuracy
MissClassificationRate_Training = 0;
MissClassificationRate_Testing = 0;
for i = 1 : size(T, 2)
    [~, label_index_expected] = max(T(:,i));
    [~, label_index_actual] = max(Y(:,i));
    if label_index_actual~=label_index_expected
        MissClassificationRate_Training = MissClassificationRate_Training+1;
    end
end
TrainingAccuracy = 1-MissClassificationRate_Training/size(T,2);
%% Test Accuracy
for i = 1 : size(TV.T, 2)
    [~, label_index_expected] = max(TV.T(:,i));
    [~, label_index_actual] = max(TY(:,i));
    if label_index_actual~=label_index_expected
        MissClassificationRate_Testing = MissClassificationRate_Testing+1;
    end
end
TestingAccuracy = 1-MissClassificationRate_Testing/size(TV.T,2);
end
%% PROCRUST Orthogonal Procrustes problem
function [A2, Q, r] = procrustNew( A, B )
% A2 = PROCRUST( A, B ) applies an orthogonal transformation to matrix B
% by multiplication with Q such that A-B*Q has minimum Frobenius norm. The
% results B*Q is returned as A2.
% [A2, Q] = PROCRUST( A, B ) also returns the orthogonal matrix Q that was
% used for the transformation.
% [A2, Q, R] = PROCRUST( A, B ) also returns the Frobenius norm of A-B*Q.
% Author : E. Larsen
% Date   : 12/22/03
% Email  : erik.larsen@ieee.org
% Reference: Golub and van Loan, p. 601.
% Error checking
msg = nargchk( 2, 2, nargin );
if ~isempty( msg )
    error( msg )
end
%% Do the computation
C = B.'*A;
[~, ~, V1] = svd(C'*C);
[U2, ~, ~] = svd(C*C');
Q = U2*V1';
A2 = B*Q;
%% Optional output of norm
if nargout > 2
    r = norm(A-A2, 'fro');
end
end
%% Accuracy 
%% Computing Accuracy
function [OA, kappa, AA, PC] = My_Accuracy(True, Predicted, uc)
nrPixelsPerClass = zeros(1,length(uc))';
Conf = zeros(length(uc),length(uc));
for l_true=1:length(uc)
    tmp_true = find (True == (l_true-1));
    nrPixelsPerClass(l_true) = length(tmp_true);
    for l_seg=1:length(uc)
        tmp_seg = find (Predicted == (l_seg-1));
        nrPixels = length(intersect(tmp_true,tmp_seg));
        Conf(l_true,l_seg) = nrPixels;  
    end
end
diagVector = diag(Conf);
PC = (diagVector./(nrPixelsPerClass));
AA = mean(PC);
OA = sum(Predicted == True)/length(True);
kappa = (sum(Conf(:))*sum(diag(Conf)) - sum(Conf)*sum(Conf,2))...
    /(sum(Conf(:))^2 -  sum(Conf)*sum(Conf,2));
end
%% Membership
function ELM_score = My_Member(uc, ELM_score)
%% Reformulate the Membership Matrix as per the defination
for r = 1:size(ELM_score,1)
    minVal = 0;
    maxVal = 1/numel(uc);
    [~, ind] = max(ELM_score(r,:));
    ELM_score(r,:) = minVal + (maxVal - minVal)*rand(1,numel(uc));
    AD1 = sum(ELM_score(r,:));
    AB1 = ELM_score(r,ind);
    AB2 = AD1 - AB1;
    AB3 = 1 - AB2;
    ELM_score(r,ind) = AB3;
end
end
%% Fuzziness
function MemberShip = My_Fuzziness(MemberShip)
%% Compute Fuzziness
Fuzziness = zeros(size(MemberShip,1),1);
% [b, c] = size(MemberShip);
for l = 1:size(Fuzziness,1)
    Fuzziness(l,:) = fuzziness(MemberShip(l,:));
end
Fuzziness = real(Fuzziness);
MemberShip = nonzeros(Fuzziness);
end
%% Fuzziness 
function E = fuzziness(u_j)
E = 0.01;
c = size(u_j,2);
flt_min = 1.175494e-38;
for i=1:c
    E = E-1/c*(u_j(1,i)*log2(u_j(1,i)+flt_min)+(1-u_j(1,i))*log2(1-u_j(1,i)+flt_min));
end
end
%% Good Luck Folks