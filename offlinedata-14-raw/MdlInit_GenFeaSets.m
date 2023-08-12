% Model initialization
%   Set parameters for data preprocessing
%   Extract data of 4 action types separately from raw data
%   generate feature sets of training, validation and test samples
%
% (10 vs. 1) * 5 times (to optimize hyperparameters of model training)
% 10 vs. 1: Training: Folds No.1-10; Validating: Fold No.11;
%           Training: Folds No.2-11; Validating: Fold No.12;
%           Training: Folds No.3-12; Validating: Fold No.13;
%           Training: Folds No.4-13; Validating: Fold No.14;
%           Training: Folds No.5-14; Validating: Fold No.15;
%
% (10 vs. 1) * 5 times (to test the performance of models)
% 10 vs. 1: Training: Folds No.6-15; Testing: Fold No.16;
%           Training: Folds No.7-16; Testing: Fold No.17;
%           Training: Folds No.8-17; Testing: Fold No.18;
%           Training: Folds No.9-18; Testing: Fold No.19;
%           Training: Folds No.10-19; Testing: Fold No.20;
%
% to avoid repetitive feature extraction step 
% which costs most computing time in searching optimized [c,g].

addpath(genpath('toolbox'));
fclose all;clc;clear;close all;

%% load offline data parameters (should be saved together with offline data)
% data sample param
sample_frequency = 256;
trials_per_action = 20;                                                    % Trial nummber/action in offline task
seconds_per_TrainTrial = 4;
data_points_per_TrainTrial = sample_frequency * seconds_per_TrainTrial;

WindowLength = 512;                                                        % Window length
SlideWindowLength = 256;                                                   % Slide length
windows_per_trial = (data_points_per_TrainTrial - WindowLength) / SlideWindowLength + 1; % = 3
% windows_per_trial = 2;
windows_per_action = windows_per_trial * trials_per_action;     
windows_per_action_tmp = 3 * trials_per_action;

% channel selection
% new cap 
% Data No.  ... 30   29   28  ...  25     24     23     22   ... 20  19  18  17  16 ...
% Real Pos. ... FC1  FCz  FC2 ... FCC3h  FCC1h  FCC2h  FCC4h ... C3  C1  Cz  C2  C4 ...
% Pos. No.  ... 5    6    7   ...  10     11     12     13   ... 15  16  17  18  19 ...
% Pos. Tag  ... F3   Fz   F4  ... FT11    FC3    FCz    FC4  ... T7  C3  Cz  C4  T8 ... 
%
% Data No.  ...  14     13     12     11   ...  8    7    6  ...
% Real Pos. ... CCP3h  CCP1h  CCP2h  CCP4h ... CP1  CPz  CP2 ...
% Pos. No.  ... 21      22     23     24   ... 27   28   29  ...
% Pos. Tag  ... CPz     CP4    M1     M2   ... Pz   P4   P8  ...

% 19 channels, Cz, C1-C4, CCP1h-CCP4h, FCC1h-FCC4h, FCz, FC1-FC2, CPz, CP1-CP2
channels = [6:8,11:14,16:20,22:25,28:30]; 
number_of_channels = length(channels);

% sub fre band selection
Wband = [[4,8];[8,13];[13,20];[13,30]];
number_bandpass_filters = size(Wband,1); 
FilterType = 'bandpass';
FilterOrder = 4;   

% choose CSP param
FilterNum = 4;% cannot be bigger than number of channels (?bandpass filters)

%% Load subject information for analysing
load('subjectinfo.mat','Subject');

for subjectnum = 1:length(Subject)
    
    offlinepath = ['.\Subject_',num2str(Subject(subjectnum).number)];
    
    save([offlinepath,'\InitializationParameters.mat'],...
                  'sample_frequency',...
                  'trials_per_action','seconds_per_TrainTrial',...
                  'WindowLength','SlideWindowLength',...
                  'channels','Wband',...
                  'FilterType','FilterOrder',...
                  'FilterNum');

              
    files = dir([offlinepath,'\Offline_EEGdata_*.mat']);

    %% Load & save baseline/training data, preprocess w/ slide window, separate into 4 action types
    load([offlinepath,'\',files(1).name],'TrialData');
    TrialDataBase = double(TrialData);
    
    Tag = TrialDataBase(end,:);
    % Idle
    [~,Index_Idle] = find(Tag==1);
    TrialDataUpdate_Idle = double(TrialDataBase(:,Index_Idle));
    % Walk
    [~,Index_Walk] = find(Tag==2);
    TrialDataUpdate_Walk = double(TrialDataBase(:,Index_Walk));
    % Ascend
    [~,Index_Ascend] = find(Tag==3);
    TrialDataUpdate_Ascend = double(TrialDataBase(:,Index_Ascend));
    % Descend
    [~,Index_Descend] = find(Tag==4);
    TrialDataUpdate_Descend = double(TrialDataBase(:,Index_Descend));

    save([offlinepath,'\TrialDataUpdate.mat'],'TrialDataUpdate_Idle',...
                                             'TrialDataUpdate_Walk',...
                                             'TrialDataUpdate_Ascend',...
                                             'TrialDataUpdate_Descend');

    % put all 2-sec windows of each action together for further update
    SlideDataUpdate_Idle = cell(1, windows_per_action);                     % windows_per_action = 60;
    SlideDataUpdate_Walk = cell(1, windows_per_action);
    SlideDataUpdate_Ascend = cell(1, windows_per_action);
    SlideDataUpdate_Descend = cell(1, windows_per_action);

    for i = 1:trials_per_action
        for j = 1:windows_per_trial
            PointStart = (i-1)*data_points_per_TrainTrial + (j-1)*SlideWindowLength;
            SlideDataUpdate_Idle{1, (i-1)*windows_per_trial+j} = TrialDataUpdate_Idle(:,PointStart + 1:PointStart + WindowLength );
            SlideDataUpdate_Walk{1, (i-1)*windows_per_trial+j} = TrialDataUpdate_Walk(:,PointStart + 1:PointStart + WindowLength );
            SlideDataUpdate_Ascend{1, (i-1)*windows_per_trial+j} = TrialDataUpdate_Ascend(:,PointStart + 1:PointStart + WindowLength );
            SlideDataUpdate_Descend{1, (i-1)*windows_per_trial+j} = TrialDataUpdate_Descend(:,PointStart + 1:PointStart + WindowLength );
        end
    end
    
    SlideSample_Idle = SlideDataUpdate_Idle;
    SlideSample_Walk = SlideDataUpdate_Walk;
    SlideSample_Ascend = SlideDataUpdate_Ascend;
    SlideSample_Descend = SlideDataUpdate_Descend;
    
    %% Get & save rejection logical vector
    rejected_sample_Idle = Subject(subjectnum).rejected_sample_Idle;
    rejected_sample_Walk = Subject(subjectnum).rejected_sample_Walk;
    rejected_sample_Ascend = Subject(subjectnum).rejected_sample_Ascend;
    rejected_sample_Descend = Subject(subjectnum).rejected_sample_Descend;

    fprintf(['Subject No.',num2str(Subject(subjectnum).number),': \n']);
    fprintf(['  These samples are deleted : \n']);
    if ~isempty(rejected_sample_Idle)
        fprintf('    Idle (%d):  %s.\n',length(rejected_sample_Idle),join(string(rejected_sample_Idle),', '));
    else
        fprintf('    Idle (0):  none.\n');
    end
    if ~isempty(rejected_sample_Walk)    
        fprintf('    Walk (%d):  %s.\n',length(rejected_sample_Walk),join(string(rejected_sample_Walk),', '));
    else
        fprintf('    Walk (0):  none.\n');
    end
    if ~isempty(rejected_sample_Ascend)        
        fprintf('    Ascend (%d):  %s.\n',length(rejected_sample_Ascend),join(string(rejected_sample_Ascend),', '));
    else
        fprintf('    Ascend (0):  none.\n');
    end
    if ~isempty(rejected_sample_Descend)        
        fprintf('    Descend (%d):  %s.\n',length(rejected_sample_Descend),join(string(rejected_sample_Descend),', '));
    else
        fprintf('    Descend (0):  none.\n');
    end
    
    rejVec_Idle = false(1,windows_per_action);
    rejVec_Walk = false(1,windows_per_action);
    rejVec_Ascend = false(1,windows_per_action);
    rejVec_Descend = false(1,windows_per_action);

    rejVec_Idle(rejected_sample_Idle) = true;
    rejVec_Walk(rejected_sample_Walk) = true;
    rejVec_Ascend(rejected_sample_Ascend) = true;
    rejVec_Descend(rejected_sample_Descend) = true;
    
    acceptVec_Idle = ~rejVec_Idle;                                          % 1*60 logical, "true(1)" for accepting
    acceptVec_Walk = ~rejVec_Walk;
    acceptVec_Ascend = ~rejVec_Ascend;
    acceptVec_Descend = ~rejVec_Descend;
    
    %% Get all filter samples
    fprintf('  划分子频带...\n');
    fprintf('  Subband filtering...\n');tic;

    % shape: (number of bandpass filters, windows per action)
    FilterSample_Idle = cell(number_bandpass_filters, windows_per_action);  % 4 Filtered Freq Bands * 60 Windows
    FilterSample_Walk = cell(number_bandpass_filters, windows_per_action);
    FilterSample_Ascend = cell(number_bandpass_filters, windows_per_action);
    FilterSample_Descend = cell(number_bandpass_filters, windows_per_action);

    SampleIndex_all = zeros(4,windows_per_action);                          % 4 rows -- 4 classes
    % actual indices of all 60 samples, 0 for rejected samples
    
    acceptIndex_Idle  = find(acceptVec_Idle  == 1);                         % indices of accepted samples
    acceptIndex_Walk  = find(acceptVec_Walk  == 1);
    acceptIndex_Ascend  = find(acceptVec_Ascend  == 1);
    acceptIndex_Descend  = find(acceptVec_Descend  == 1);
    
    acceptednum_Idle = length(acceptIndex_Idle);                            % number of accepted samples
    acceptednum_Walk = length(acceptIndex_Walk);
    acceptednum_Ascend = length(acceptIndex_Ascend);
    acceptednum_Descend = length(acceptIndex_Descend);
    
    % Filter samples only those accepted
    for i = 1:acceptednum_Idle  
        for j = 1:number_bandpass_filters
            FilterSample_Idle{j,acceptIndex_Idle(i)} = Rsx_ButterFilter(FilterOrder,Wband(j,:),sample_frequency,FilterType,SlideSample_Idle{1,acceptIndex_Idle(i)}(channels,:),number_of_channels);
        end
        SampleIndex_all(1,acceptIndex_Idle(i)) = acceptIndex_Idle(i);
    end
    for i = 1:acceptednum_Walk
        for j = 1:number_bandpass_filters
            FilterSample_Walk{j,acceptIndex_Walk(i)} = Rsx_ButterFilter(FilterOrder,Wband(j,:),sample_frequency,FilterType,SlideSample_Walk{1,acceptIndex_Walk(i)}(channels,:),number_of_channels);
        end
        SampleIndex_all(2,acceptIndex_Walk(i)) = acceptIndex_Walk(i);
    end
    for i = 1:acceptednum_Ascend
        for j = 1:number_bandpass_filters
            FilterSample_Ascend{j,acceptIndex_Ascend(i)} = Rsx_ButterFilter(FilterOrder,Wband(j,:),sample_frequency,FilterType,SlideSample_Ascend{1,acceptIndex_Ascend(i)}(channels,:),number_of_channels);
        end
        SampleIndex_all(3,acceptIndex_Ascend(i)) = acceptIndex_Ascend(i);
    end
    for i = 1:acceptednum_Descend
        for j = 1:number_bandpass_filters
            FilterSample_Descend{j,acceptIndex_Descend(i)} = Rsx_ButterFilter(FilterOrder,Wband(j,:),sample_frequency,FilterType,SlideSample_Descend{1,acceptIndex_Descend(i)}(channels,:),number_of_channels);
        end
        SampleIndex_all(4,acceptIndex_Descend(i)) = acceptIndex_Descend(i);
    end
    
    toc;
    %% Make 5 training-validating, training-testing pairs of sample sets, 
    % 10/20 training (for hyperpara training), 1/20 (No.11,12,13,14,15)validating
    % 10/20 training (for mdl training), 1/20 (No.16,17,18,19,20)testing
    % in time sequence
    foldnum = 20;                                                           % 1 trial per fold
    windows_per_fold = windows_per_action/foldnum;                          % 20 * 3 / 20

    for valcount = 1:5
    %%% valcount = 1: NO.1-10 training (for val), NO.11 validating
    %%%               NO.6-15 training (for test), NO.16 test
    %%% valcount = 2: NO.2-11 training (for val), NO.12 validating
    %%%               NO.7-16 training (for test), NO.17 test
    %%% valcount = 3: NO.3-12 training (for val), NO.13 validating
    %%%               NO.8-17 training (for test), NO.18 test
    %%% valcount = 4: NO.4-13 training (for val), NO.14 validating
    %%%               NO.9-18 training (for test), NO.19 test
    %%% valcount = 5: NO.5-14 training (for val), NO.15 validating
    %%%               NO.10-19 training (for test), NO.20 test
        fprintf(['  Generating ',num2str(valcount),' feature sets...\n']);tic;

        valindex = foldnum - 10 + valcount;
        testindex = foldnum - 5 + valcount;
        samplenum_VAL_train = (foldnum - 10) * windows_per_fold;
        samplenum_TEST_train = (foldnum - 10) * windows_per_fold;
        samplenum_VAL_val = 4 * windows_per_fold;
        samplenum_TEST_test = 4 * windows_per_fold;

        %% Generate training-validation feature sets (for VAL process)
        %% Training set filter sample (for VAL process)           % _VAL_train
        SampleIndex_VAL_train = SampleIndex_all(:,(valcount-1)* windows_per_fold + 1:(valindex - 1) * windows_per_fold);
        % actual indices of _VAL_train samples, 0 for rejected samples
        
        % Get indices of replacing samples for rejected samples
        for classtype = 1:4
            for sampleIndex = 1:samplenum_VAL_train
                if SampleIndex_VAL_train(classtype,sampleIndex) == 0
                    trialIndex = floor((sampleIndex-0.1)/3);
                    sampleIndex_1 = trialIndex * 3 + 1;
                    sampleIndex_2 = trialIndex * 3 + 2;
                    sampleIndex_3 = trialIndex * 3 + 3;
                    sampleIndex_tmp = SampleIndex_VAL_train(classtype,sampleIndex_1:sampleIndex_3);
                    sampleIndex_replace = sampleIndex_tmp(sampleIndex_tmp ~= 0);
                    if ~isempty(sampleIndex_replace)
                        samplenum_replace = length(sampleIndex_replace);
                        SampleIndex_VAL_train(classtype,sampleIndex) = sampleIndex_replace(mod(sampleIndex,samplenum_replace)+1);
                    else
                        trialIndex = abs(trialIndex - 1);
                        sampleIndex_1 = trialIndex * 3 + 1;
                        sampleIndex_2 = trialIndex * 3 + 2;
                        sampleIndex_3 = trialIndex * 3 + 3;
                        sampleIndex_tmp = SampleIndex_VAL_train(classtype,sampleIndex_1:sampleIndex_3);
                        sampleIndex_replace = sampleIndex_tmp(sampleIndex_tmp ~= 0);
                        samplenum_replace = length(sampleIndex_replace);
                        SampleIndex_VAL_train(classtype,sampleIndex) = sampleIndex_replace(mod(0,samplenum_replace)+1);
                        SampleIndex_VAL_train(classtype,sampleIndex+1) = sampleIndex_replace(mod(1,samplenum_replace)+1);
                        SampleIndex_VAL_train(classtype,sampleIndex+2) = sampleIndex_replace(mod(2,samplenum_replace)+1);
                    end
                end
            end
        end

        FilterSample_Idle_VAL_train = FilterSample_Idle(:,SampleIndex_VAL_train(1,:));
        FilterSample_Walk_VAL_train = FilterSample_Walk(:,SampleIndex_VAL_train(2,:));
        FilterSample_Ascend_VAL_train = FilterSample_Ascend(:,SampleIndex_VAL_train(3,:));
        FilterSample_Descend_VAL_train = FilterSample_Descend(:,SampleIndex_VAL_train(4,:));
 
        FilterSample_VAL_train_isIdle = repmat(FilterSample_Idle_VAL_train,1,3);
        FilterSample_VAL_train_notIdle = [FilterSample_Walk_VAL_train,...
                                          FilterSample_Ascend_VAL_train,...
                                          FilterSample_Descend_VAL_train];
                                      
        %% Validation set filter sample (for VAL process)         % _VAL_val
        FilterSample_VAL_val_init = [FilterSample_Idle(:,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold), ...
                                     FilterSample_Walk(:,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold), ...
                                     FilterSample_Ascend(:,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold), ...
                                     FilterSample_Descend(:,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold)];
        % [] for rejected sample
        FilterSample_Y_VAL_val = [ones(windows_per_fold,1);ones(windows_per_fold,1)*2;...
                                  ones(windows_per_fold,1)*3;ones(windows_per_fold,1)*4];
                              
        SampleIndex_VAL_val = [SampleIndex_all(1,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold),...
                               SampleIndex_all(2,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold),...
                               SampleIndex_all(3,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold),...
                               SampleIndex_all(4,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold)];
        % actual indices of _VAL_val samples, 0 for rejected samples
        
        % Get indices of replacing samples for rejected samples
        % Fill replacing samples in []s for rejected samples
        FilterSample_VAL_val = cell(size(FilterSample_VAL_val_init));
        for sampleIndex = 1:samplenum_VAL_val
            if SampleIndex_VAL_val(sampleIndex) ~= 0
                FilterSample_VAL_val(:,sampleIndex) = FilterSample_VAL_val_init(:,sampleIndex);
            else
                trialIndex = floor((sampleIndex-0.1)/3);
                sampleIndex_1 = trialIndex * 3 + 1;
                sampleIndex_2 = trialIndex * 3 + 2;
                sampleIndex_3 = trialIndex * 3 + 3;
                sampleIndex_tmp = SampleIndex_VAL_val(sampleIndex_1:sampleIndex_3);
                sampleIndex_replace = sampleIndex_tmp(sampleIndex_tmp ~= 0);
                FilterSample_Val_tmp = FilterSample_VAL_val_init(:,sampleIndex_1:sampleIndex_3);
                FilterSample_Val_replace = FilterSample_Val_tmp(:,(sampleIndex_tmp ~= 0));
                if ~isempty(sampleIndex_replace)
                    SampleIndex_VAL_val(sampleIndex) = sampleIndex_replace(1);
                    FilterSample_VAL_val(:,sampleIndex) = FilterSample_Val_replace(:,1);
                else
                    FilterSample_VAL_val(:,sampleIndex) = [];
                end
            end
        end
        
        %% Geting CSP for VAL process
        fprintf('    生成验证过程csp矩阵...\n');
        fprintf('    Generating CSP matrices for Validation process...\n');
        CspTranspose_VAL_1 = cell(1,number_bandpass_filters);
        CspTranspose_VAL_23 = cell(1,number_bandpass_filters);
        CspTranspose_VAL_24 = cell(1,number_bandpass_filters);
        CspTranspose_VAL_34 = cell(1,number_bandpass_filters);
        for i = 1:number_bandpass_filters
            CspTranspose_VAL_1{i} = Rsx_CSP_R3(FilterSample_VAL_train_isIdle(i,:),FilterSample_VAL_train_notIdle(i,:)); 
            CspTranspose_VAL_23{i} = Rsx_CSP_R3(FilterSample_Walk_VAL_train(i,:),FilterSample_Ascend_VAL_train(i,:)); 
            CspTranspose_VAL_24{i} = Rsx_CSP_R3(FilterSample_Walk_VAL_train(i,:),FilterSample_Descend_VAL_train(i,:)); 
            CspTranspose_VAL_34{i} = Rsx_CSP_R3(FilterSample_Ascend_VAL_train(i,:),FilterSample_Descend_VAL_train(i,:));  
        end
        CSPs_VAL = cell(1,4);
        CSPs_VAL{1} = CspTranspose_VAL_1;
        CSPs_VAL{2} = CspTranspose_VAL_23;
        CSPs_VAL{3} = CspTranspose_VAL_24;
        CSPs_VAL{4} = CspTranspose_VAL_34;

        %% CSP Feature extraction for generating classifier (For VAL process)
        % CSP Feature extraction (of training samples)
        fprintf('    生成验证过程训练集特征及标签...\n');
        fprintf('    CSP Feature extraction for generating classifier (for Validation process)...\n');
        % Idle vs. notIdle
        % shape: (samplenum, number of bandpass filters * FilterNum)
        FeaSample_isIdle_VAL_train_1 = [];
        FeaSample_notIdle_VAL_train_1 = [];
        for i = 1:samplenum_VAL_train*3                                     %每个window的每个freq band进行特征提取
            FeaTemp_isIdle = [];
            FeaTemp_notIdle = [];
            for j =1:number_bandpass_filters
                FeaTemp_isIdle = [FeaTemp_isIdle,Rsx_singlewindow_cspfeature(FilterSample_VAL_train_isIdle{j,i},CspTranspose_VAL_1{j},FilterNum)];
                FeaTemp_notIdle = [FeaTemp_notIdle,Rsx_singlewindow_cspfeature(FilterSample_VAL_train_notIdle{j,i},CspTranspose_VAL_1{j},FilterNum)];
            end
            FeaSample_isIdle_VAL_train_1 = [FeaSample_isIdle_VAL_train_1;FeaTemp_isIdle];
            FeaSample_notIdle_VAL_train_1 = [FeaSample_notIdle_VAL_train_1;FeaTemp_notIdle];
        end

        % Walk vs. Ascend
        % shape: (samplenum, number of bandpass filters * FilterNum)
        FeaSample_Walk_VAL_train_23 = [];
        FeaSample_Ascend_VAL_train_23 = [];
        for i = 1:samplenum_VAL_train
            FeaTemp_Walk = [];
            FeaTemp_Ascend = [];
            for j =1:number_bandpass_filters
                FeaTemp_Walk = [FeaTemp_Walk,Rsx_singlewindow_cspfeature(FilterSample_Walk_VAL_train{j,i},CspTranspose_VAL_23{j},FilterNum)];
                FeaTemp_Ascend = [FeaTemp_Ascend,Rsx_singlewindow_cspfeature(FilterSample_Ascend_VAL_train{j,i},CspTranspose_VAL_23{j},FilterNum)];
            end
            FeaSample_Walk_VAL_train_23 = [FeaSample_Walk_VAL_train_23;FeaTemp_Walk];
            FeaSample_Ascend_VAL_train_23 = [FeaSample_Ascend_VAL_train_23;FeaTemp_Ascend];
        end

        % Walk vs. Descend
        % shape: (samplenum, number of bandpass filters * FilterNum)
        FeaSample_Walk_VAL_train_24 = [];
        FeaSample_Descend_VAL_train_24 = [];
        for i = 1:samplenum_VAL_train
            FeaTemp_Walk = [];
            FeaTemp_Descend = [];
            for j =1:number_bandpass_filters
                FeaTemp_Walk = [FeaTemp_Walk,Rsx_singlewindow_cspfeature(FilterSample_Walk_VAL_train{j,i},CspTranspose_VAL_24{j},FilterNum)];
                FeaTemp_Descend = [FeaTemp_Descend,Rsx_singlewindow_cspfeature(FilterSample_Descend_VAL_train{j,i},CspTranspose_VAL_24{j},FilterNum)];
            end
            FeaSample_Walk_VAL_train_24 = [FeaSample_Walk_VAL_train_24;FeaTemp_Walk];
            FeaSample_Descend_VAL_train_24 = [FeaSample_Descend_VAL_train_24;FeaTemp_Descend];
        end

        % Ascend vs. Descend
        % shape: (samplenum, number of bandpass filters * FilterNum)
        FeaSample_Ascend_VAL_train_34 = [];
        FeaSample_Descend_VAL_train_34 = [];
        for i = 1:samplenum_VAL_train
            FeaTemp_Ascend = [];
            FeaTemp_Descend = [];
            for j =1:number_bandpass_filters
                FeaTemp_Ascend = [FeaTemp_Ascend,Rsx_singlewindow_cspfeature(FilterSample_Ascend_VAL_train{j,i},CspTranspose_VAL_34{j},FilterNum)];
                FeaTemp_Descend = [FeaTemp_Descend,Rsx_singlewindow_cspfeature(FilterSample_Descend_VAL_train{j,i},CspTranspose_VAL_34{j},FilterNum)];
            end
            FeaSample_Ascend_VAL_train_34 = [FeaSample_Ascend_VAL_train_34;FeaTemp_Ascend];
            FeaSample_Descend_VAL_train_34 = [FeaSample_Descend_VAL_train_34;FeaTemp_Descend];
        end

        %% Min-max normalization
        % Idle vs. notIdle
        VAL_train_Fea_1_init = [FeaSample_isIdle_VAL_train_1;FeaSample_notIdle_VAL_train_1]; 
        VAL_train_Fea_Y_1 = [ones(samplenum_VAL_train*3,1);ones(samplenum_VAL_train*3,1)*2];
        FeaMin_VAL_1 = min(VAL_train_Fea_1_init);
        FeaMax_VAL_1 = max(VAL_train_Fea_1_init);
        VAL_train_Fea_1 = (VAL_train_Fea_1_init - FeaMin_VAL_1)./(FeaMax_VAL_1-FeaMin_VAL_1);

        % Walk vs. Ascend
        VAL_train_Fea_23_init = [FeaSample_Walk_VAL_train_23;FeaSample_Ascend_VAL_train_23];
        VAL_train_Fea_Y_23 = [ones(samplenum_VAL_train,1);ones(samplenum_VAL_train,1)*2];
        FeaMin_VAL_23 = min(VAL_train_Fea_23_init);
        FeaMax_VAL_23 = max(VAL_train_Fea_23_init);
        VAL_train_Fea_23 = (VAL_train_Fea_23_init - FeaMin_VAL_23)./(FeaMax_VAL_23-FeaMin_VAL_23);

        % Walk vs. Descend
        VAL_train_Fea_24_init = [FeaSample_Walk_VAL_train_24;FeaSample_Descend_VAL_train_24];
        VAL_train_Fea_Y_24 = [ones(samplenum_VAL_train,1);ones(samplenum_VAL_train,1)*2];
        FeaMin_VAL_24 = min(VAL_train_Fea_24_init);
        FeaMax_VAL_24 = max(VAL_train_Fea_24_init);
        VAL_train_Fea_24 = (VAL_train_Fea_24_init - FeaMin_VAL_24)./(FeaMax_VAL_24-FeaMin_VAL_24);

        % Ascend vs. Descend
        VAL_train_Fea_34_init = [FeaSample_Ascend_VAL_train_34;FeaSample_Descend_VAL_train_34];
        VAL_train_Fea_Y_34 = [ones(samplenum_VAL_train,1);ones(samplenum_VAL_train,1)*2];
        FeaMin_VAL_34 = min(VAL_train_Fea_34_init);
        FeaMax_VAL_34 = max(VAL_train_Fea_34_init);
        VAL_train_Fea_34 = (VAL_train_Fea_34_init - FeaMin_VAL_34)./(FeaMax_VAL_34-FeaMin_VAL_34);

        %% CSP Feature extraction for classifier validation (For VAL process)
        % CSP Feature extraction (of validating samples)
        fprintf('    生成验证集特征及标签...\n');
        fprintf('    CSP Feature extraction for classifier validation (For Validation process)...\n');
        samplenum_VAL_val = sum(SampleIndex_VAL_val ~= 0);                  % in case a whole trial is rejected
        acceptIndex_VAL_val = find(SampleIndex_VAL_val ~= 0);
        VAL_val_Fea_1 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
        VAL_val_Fea_23 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
        VAL_val_Fea_24 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
        VAL_val_Fea_34 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
        for i = 1:samplenum_VAL_val                                         %每个window的每个freq band进行特征提取
            FeaTemp_1 = [];
            FeaTemp_23 = [];
            FeaTemp_24 = [];
            FeaTemp_34 = [];
            for j =1:number_bandpass_filters
                FeaTemp_1 = [FeaTemp_1,Rsx_singlewindow_cspfeature(FilterSample_VAL_val{j,acceptIndex_VAL_val(i)},CspTranspose_VAL_1{j},FilterNum)];
                FeaTemp_23 = [FeaTemp_23,Rsx_singlewindow_cspfeature(FilterSample_VAL_val{j,acceptIndex_VAL_val(i)},CspTranspose_VAL_23{j},FilterNum)];
                FeaTemp_24 = [FeaTemp_24,Rsx_singlewindow_cspfeature(FilterSample_VAL_val{j,acceptIndex_VAL_val(i)},CspTranspose_VAL_24{j},FilterNum)];
                FeaTemp_34 = [FeaTemp_34,Rsx_singlewindow_cspfeature(FilterSample_VAL_val{j,acceptIndex_VAL_val(i)},CspTranspose_VAL_34{j},FilterNum)];
            end
            VAL_val_Fea_1(acceptIndex_VAL_val(i),:) = (FeaTemp_1 - FeaMin_VAL_1)./(FeaMax_VAL_1 - FeaMin_VAL_1);
            VAL_val_Fea_23(acceptIndex_VAL_val(i),:) = (FeaTemp_23 - FeaMin_VAL_23)./(FeaMax_VAL_23 - FeaMin_VAL_23);
            VAL_val_Fea_24(acceptIndex_VAL_val(i),:) = (FeaTemp_24 - FeaMin_VAL_24)./(FeaMax_VAL_24-FeaMin_VAL_24);
            VAL_val_Fea_34(acceptIndex_VAL_val(i),:) = (FeaTemp_34 - FeaMin_VAL_34)./(FeaMax_VAL_34-FeaMin_VAL_34);
        end
        VAL_val_Fea_all = cell(1,4);
        VAL_val_Fea_all{1} = VAL_val_Fea_1;
        VAL_val_Fea_all{2} = VAL_val_Fea_23;
        VAL_val_Fea_all{3} = VAL_val_Fea_24;
        VAL_val_Fea_all{4} = VAL_val_Fea_34;
        VAL_val_Fea_Y = FilterSample_Y_VAL_val;

        
        %% Generate training-testing feature sets (for TEST process)
        %% Training set filter sample (for TEST process)          % _TEST_train
        SampleIndex_TEST_train = SampleIndex_all(:,(valcount+4)* windows_per_fold + 1:(testindex - 1) * windows_per_fold);
        
        for classtype = 1:4
            for sampleIndex = 1:samplenum_TEST_train
                if SampleIndex_TEST_train(classtype,sampleIndex) == 0
                    trialIndex = floor((sampleIndex-0.1)/3);
                    sampleIndex_1 = trialIndex * 3 + 1;
                    sampleIndex_2 = trialIndex * 3 + 2;
                    sampleIndex_3 = trialIndex * 3 + 3;
                    sampleIndex_tmp = SampleIndex_TEST_train(classtype,sampleIndex_1:sampleIndex_3);
                    sampleIndex_replace = sampleIndex_tmp(sampleIndex_tmp ~= 0);
                    if ~isempty(sampleIndex_replace)
                        samplenum_replace = length(sampleIndex_replace);
                        SampleIndex_TEST_train(classtype,sampleIndex) = sampleIndex_replace(mod(sampleIndex,samplenum_replace)+1);
                    else
                        trialIndex = abs(trialIndex - 1);
                        sampleIndex_1 = trialIndex * 3 + 1;
                        sampleIndex_2 = trialIndex * 3 + 2;
                        sampleIndex_3 = trialIndex * 3 + 3;
                        sampleIndex_tmp = SampleIndex_TEST_train(classtype,sampleIndex_1:sampleIndex_3);
                        sampleIndex_replace = sampleIndex_tmp(sampleIndex_tmp ~= 0);
                        samplenum_replace = length(sampleIndex_replace);
                        SampleIndex_TEST_train(classtype,sampleIndex) = sampleIndex_replace(mod(0,samplenum_replace)+1);
                        SampleIndex_TEST_train(classtype,sampleIndex+1) = sampleIndex_replace(mod(1,samplenum_replace)+1);
                        SampleIndex_TEST_train(classtype,sampleIndex+2) = sampleIndex_replace(mod(2,samplenum_replace)+1);
                    end
                end
            end
        end

        FilterSample_Idle_TEST_train = FilterSample_Idle(:,SampleIndex_TEST_train(1,:));
        FilterSample_Walk_TEST_train = FilterSample_Walk(:,SampleIndex_TEST_train(2,:));
        FilterSample_Ascend_TEST_train = FilterSample_Ascend(:,SampleIndex_TEST_train(3,:));
        FilterSample_Descend_TEST_train = FilterSample_Descend(:,SampleIndex_TEST_train(4,:));
 
        FilterSample_TEST_train_isIdle = repmat(FilterSample_Idle_TEST_train,1,3);
        FilterSample_TEST_train_notIdle = [FilterSample_Walk_TEST_train,...
                                          FilterSample_Ascend_TEST_train,...
                                          FilterSample_Descend_TEST_train];

        %% Test set filter sample (for TEST process)              % _TEST_test
        FilterSample_TEST_test_init = [FilterSample_Idle(:,((testindex - 1) * windows_per_fold + 1):testindex * windows_per_fold), ...
                                  FilterSample_Walk(:,((testindex - 1) * windows_per_fold + 1):testindex * windows_per_fold), ...
                                  FilterSample_Ascend(:,((testindex - 1) * windows_per_fold + 1):testindex * windows_per_fold), ...
                                  FilterSample_Descend(:,((testindex - 1) * windows_per_fold + 1):testindex * windows_per_fold)];
        FilterSample_Y_TEST_test = [ones(windows_per_fold,1);ones(windows_per_fold,1)*2;...
                                    ones(windows_per_fold,1)*3;ones(windows_per_fold,1)*4];

        SampleIndex_TEST_test = [SampleIndex_all(1,((testindex - 1) * windows_per_fold + 1):testindex * windows_per_fold),...
                                 SampleIndex_all(2,((testindex - 1) * windows_per_fold + 1):testindex * windows_per_fold),...
                                 SampleIndex_all(3,((testindex - 1) * windows_per_fold + 1):testindex * windows_per_fold),...
                                 SampleIndex_all(4,((testindex - 1) * windows_per_fold + 1):testindex * windows_per_fold)];
                           
        FilterSample_TEST_test = cell(size(FilterSample_TEST_test_init));
        for sampleIndex = 1:samplenum_TEST_test
            if SampleIndex_TEST_test(sampleIndex) ~= 0
                FilterSample_TEST_test(:,sampleIndex) = FilterSample_TEST_test_init(:,sampleIndex);
            else
                trialIndex = floor((sampleIndex-0.1)/3);
                sampleIndex_1 = trialIndex * 3 + 1;
                sampleIndex_2 = trialIndex * 3 + 2;
                sampleIndex_3 = trialIndex * 3 + 3;
                sampleIndex_tmp = SampleIndex_TEST_test(sampleIndex_1:sampleIndex_3);
                sampleIndex_replace = sampleIndex_tmp(sampleIndex_tmp ~= 0);
                FilterSample_Test_tmp = FilterSample_TEST_test_init(:,sampleIndex_1:sampleIndex_3);
                FilterSample_Test_replace = FilterSample_Test_tmp(:,(sampleIndex_tmp ~= 0));
                if ~isempty(sampleIndex_replace)
                    SampleIndex_TEST_test(sampleIndex) = sampleIndex_replace(1);
                    FilterSample_TEST_test(:,sampleIndex) = FilterSample_Test_replace(:,1);
                else
                    FilterSample_TEST_test(:,sampleIndex) = [];
                end
            end
        end
        
        %% Geting CSP for TEST process
        fprintf('    生成测试过程csp矩阵...\n');
        fprintf('    Generating CSP matrices for Test process...\n');
        CspTranspose_TEST_1 = cell(1,number_bandpass_filters);
        CspTranspose_TEST_23 = cell(1,number_bandpass_filters);
        CspTranspose_TEST_24 = cell(1,number_bandpass_filters);
        CspTranspose_TEST_34 = cell(1,number_bandpass_filters);
        for i = 1:number_bandpass_filters
            CspTranspose_TEST_1{i} = Rsx_CSP_R3(FilterSample_TEST_train_isIdle(i,:),FilterSample_TEST_train_notIdle(i,:)); 
            CspTranspose_TEST_23{i} = Rsx_CSP_R3(FilterSample_Walk_TEST_train(i,:),FilterSample_Ascend_TEST_train(i,:)); 
            CspTranspose_TEST_24{i} = Rsx_CSP_R3(FilterSample_Walk_TEST_train(i,:),FilterSample_Descend_TEST_train(i,:)); 
            CspTranspose_TEST_34{i} = Rsx_CSP_R3(FilterSample_Ascend_TEST_train(i,:),FilterSample_Descend_TEST_train(i,:));  
        end
        CSPs_TEST = cell(1,4);
        CSPs_TEST{1} = CspTranspose_TEST_1;
        CSPs_TEST{2} = CspTranspose_TEST_23;
        CSPs_TEST{3} = CspTranspose_TEST_24;
        CSPs_TEST{4} = CspTranspose_TEST_34;

        %% CSP Feature extraction for generating classifier (for TEST process)
        % CSP Feature extraction (of training samples)
        fprintf('    生成测试过程训练集特征及标签...\n');
        fprintf('    CSP Feature extraction for generating classifier (for Test process)...\n');
        % Idle vs. notIdle
        % shape: (samplenum, number of bandpass filters * FilterNum)
        FeaSample_isIdle_TEST_train_1 = [];
        FeaSample_notIdle_TEST_train_1 = [];
        for i = 1:samplenum_TEST_train*3 %每个window的每个freq band进行特征提取
            FeaTemp_isIdle_TEST = [];
            FeaTemp_notIdle_TEST = [];
            for j =1:number_bandpass_filters
                FeaTemp_isIdle_TEST = [FeaTemp_isIdle_TEST,Rsx_singlewindow_cspfeature(FilterSample_TEST_train_isIdle{j,i},CspTranspose_TEST_1{j},FilterNum)];
                FeaTemp_notIdle_TEST = [FeaTemp_notIdle_TEST,Rsx_singlewindow_cspfeature(FilterSample_TEST_train_notIdle{j,i},CspTranspose_TEST_1{j},FilterNum)];
            end
            FeaSample_isIdle_TEST_train_1 = [FeaSample_isIdle_TEST_train_1;FeaTemp_isIdle_TEST];
            FeaSample_notIdle_TEST_train_1 = [FeaSample_notIdle_TEST_train_1;FeaTemp_notIdle_TEST];
        end

        % Walk vs. Ascend
        % shape: (samplenum, number of bandpass filters * FilterNum)
        FeaSample_Walk_TEST_train_23 = [];
        FeaSample_Ascend_TEST_train_23 = [];
        for i = 1:samplenum_TEST_train %每个window的每个freq band进行特征提取
            FeaTemp_Walk_TEST = [];
            FeaTemp_Ascend_TEST = [];
            for j =1:number_bandpass_filters
                FeaTemp_Walk_TEST = [FeaTemp_Walk_TEST,Rsx_singlewindow_cspfeature(FilterSample_Walk_TEST_train{j,i},CspTranspose_TEST_23{j},FilterNum)];
                FeaTemp_Ascend_TEST = [FeaTemp_Ascend_TEST,Rsx_singlewindow_cspfeature(FilterSample_Ascend_TEST_train{j,i},CspTranspose_TEST_23{j},FilterNum)];
            end
            FeaSample_Walk_TEST_train_23 = [FeaSample_Walk_TEST_train_23;FeaTemp_Walk_TEST];
            FeaSample_Ascend_TEST_train_23 = [FeaSample_Ascend_TEST_train_23;FeaTemp_Ascend_TEST];
        end

        % Walk vs. Descend
        % shape: (samplenum, number of bandpass filters * FilterNum)
        FeaSample_Walk_TEST_train_24 = [];
        FeaSample_Descend_TEST_train_24 = [];
        for i = 1:samplenum_TEST_train %每个window的每个freq band进行特征提取
            FeaTemp_Walk_TEST = [];
            FeaTemp_Descend_TEST = [];
            for j =1:number_bandpass_filters
                FeaTemp_Walk_TEST = [FeaTemp_Walk_TEST,Rsx_singlewindow_cspfeature(FilterSample_Walk_TEST_train{j,i},CspTranspose_TEST_24{j},FilterNum)];
                FeaTemp_Descend_TEST = [FeaTemp_Descend_TEST,Rsx_singlewindow_cspfeature(FilterSample_Descend_TEST_train{j,i},CspTranspose_TEST_24{j},FilterNum)];
            end
            FeaSample_Walk_TEST_train_24 = [FeaSample_Walk_TEST_train_24;FeaTemp_Walk_TEST];
            FeaSample_Descend_TEST_train_24 = [FeaSample_Descend_TEST_train_24;FeaTemp_Descend_TEST];
        end

        % Ascend vs. Descend
        % shape: (samplenum, number of bandpass filters * FilterNum)
        FeaSample_Ascend_TEST_train_34 = [];
        FeaSample_Descend_TEST_train_34 = [];
        for i = 1:samplenum_TEST_train %每个window的每个freq band进行特征提取
            FeaTemp_Ascend_TEST = [];
            FeaTemp_Descend_TEST = [];
            for j =1:number_bandpass_filters
                FeaTemp_Ascend_TEST = [FeaTemp_Ascend_TEST,Rsx_singlewindow_cspfeature(FilterSample_Ascend_TEST_train{j,i},CspTranspose_TEST_34{j},FilterNum)];
                FeaTemp_Descend_TEST = [FeaTemp_Descend_TEST,Rsx_singlewindow_cspfeature(FilterSample_Descend_TEST_train{j,i},CspTranspose_TEST_34{j},FilterNum)];
            end
            FeaSample_Ascend_TEST_train_34 = [FeaSample_Ascend_TEST_train_34;FeaTemp_Ascend_TEST];
            FeaSample_Descend_TEST_train_34 = [FeaSample_Descend_TEST_train_34;FeaTemp_Descend_TEST];
        end

        %% Min-max normalization
        % Idle vs. notIdle
        TEST_train_Fea_1_init = [FeaSample_isIdle_TEST_train_1;FeaSample_notIdle_TEST_train_1]; 
        TEST_train_Fea_Y_1 = [ones(samplenum_TEST_train*3,1);ones(samplenum_TEST_train*3,1)*2];
        FeaMin_TEST_1 = min(TEST_train_Fea_1_init);
        FeaMax_TEST_1 = max(TEST_train_Fea_1_init);
        TEST_train_Fea_1 = (TEST_train_Fea_1_init - FeaMin_TEST_1)./(FeaMax_TEST_1-FeaMin_TEST_1);

        % Walk vs. Ascend
        TEST_train_Fea_23_init = [FeaSample_Walk_TEST_train_23;FeaSample_Ascend_TEST_train_23];
        TEST_train_Fea_Y_23 = [ones(samplenum_TEST_train,1);ones(samplenum_TEST_train,1)*2];
        FeaMin_TEST_23 = min(TEST_train_Fea_23_init);
        FeaMax_TEST_23 = max(TEST_train_Fea_23_init);
        TEST_train_Fea_23 = (TEST_train_Fea_23_init - FeaMin_TEST_23)./(FeaMax_TEST_23-FeaMin_TEST_23);

        % Walk vs. Descend
        TEST_train_Fea_24_init = [FeaSample_Walk_TEST_train_24;FeaSample_Descend_TEST_train_24];
        TEST_train_Fea_Y_24 = [ones(samplenum_TEST_train,1);ones(samplenum_TEST_train,1)*2];
        FeaMin_TEST_24 = min(TEST_train_Fea_24_init);
        FeaMax_TEST_24 = max(TEST_train_Fea_24_init);
        TEST_train_Fea_24 = (TEST_train_Fea_24_init - FeaMin_TEST_24)./(FeaMax_TEST_24-FeaMin_TEST_24);

        % Ascend vs. Descend
        TEST_train_Fea_34_init = [FeaSample_Ascend_TEST_train_34;FeaSample_Descend_TEST_train_34];
        TEST_train_Fea_Y_34 = [ones(samplenum_TEST_train,1);ones(samplenum_TEST_train,1)*2];
        FeaMin_TEST_34 = min(TEST_train_Fea_34_init);
        FeaMax_TEST_34 = max(TEST_train_Fea_34_init);
        TEST_train_Fea_34 = (TEST_train_Fea_34_init - FeaMin_TEST_34)./(FeaMax_TEST_34-FeaMin_TEST_34);

        %% CSP Feature extraction for classifier test (For TEST process)
        % CSP Feature extraction (of testing samples)
        fprintf('    生成测试集特征及标签...\n');
        fprintf('    CSP Feature extraction for classifier test (For Test process)...\n');
        samplenum_TEST_test = sum(SampleIndex_TEST_test ~= 0);
        acceptIndex_TEST_test = find(SampleIndex_TEST_test ~= 0);
        TEST_test_Fea_1 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
        TEST_test_Fea_23 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
        TEST_test_Fea_24 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
        TEST_test_Fea_34 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
        for i = 1:samplenum_TEST_test %每个window的每个freq band进行特征提取
            FeaTemp_1 = [];
            FeaTemp_23 = [];
            FeaTemp_24 = [];
            FeaTemp_34 = [];
            for j =1:number_bandpass_filters
                FeaTemp_1 = [FeaTemp_1,Rsx_singlewindow_cspfeature(FilterSample_TEST_test{j,acceptIndex_TEST_test(i)},CspTranspose_TEST_1{j},FilterNum)];
                FeaTemp_23 = [FeaTemp_23,Rsx_singlewindow_cspfeature(FilterSample_TEST_test{j,acceptIndex_TEST_test(i)},CspTranspose_TEST_23{j},FilterNum)];
                FeaTemp_24 = [FeaTemp_24,Rsx_singlewindow_cspfeature(FilterSample_TEST_test{j,acceptIndex_TEST_test(i)},CspTranspose_TEST_24{j},FilterNum)];
                FeaTemp_34 = [FeaTemp_34,Rsx_singlewindow_cspfeature(FilterSample_TEST_test{j,acceptIndex_TEST_test(i)},CspTranspose_TEST_34{j},FilterNum)];
            end
            TEST_test_Fea_1(acceptIndex_TEST_test(i),:) = (FeaTemp_1 - FeaMin_TEST_1)./(FeaMax_TEST_1-FeaMin_TEST_1);
            TEST_test_Fea_23(acceptIndex_TEST_test(i),:) = (FeaTemp_23 - FeaMin_TEST_23)./(FeaMax_TEST_23-FeaMin_TEST_23);
            TEST_test_Fea_24(acceptIndex_TEST_test(i),:) = (FeaTemp_24 - FeaMin_TEST_24)./(FeaMax_TEST_24-FeaMin_TEST_24);
            TEST_test_Fea_34(acceptIndex_TEST_test(i),:) = (FeaTemp_34 - FeaMin_TEST_34)./(FeaMax_TEST_34-FeaMin_TEST_34);
        end
        TEST_test_Fea_all = cell(1,4);
        TEST_test_Fea_all{1} = TEST_test_Fea_1;
        TEST_test_Fea_all{2} = TEST_test_Fea_23;
        TEST_test_Fea_all{3} = TEST_test_Fea_24;
        TEST_test_Fea_all{4} = TEST_test_Fea_34;
        TEST_test_Fea_Y = FilterSample_Y_TEST_test;
        
        toc;
        
        
        %% Save feature sets (both for VAL process & TEST process)
        save([offlinepath,'\FeaSets_',num2str(valcount),'_for_classifiers_trte.mat'],...
                'VAL_train_Fea_1','VAL_train_Fea_Y_1',...
                'VAL_train_Fea_23','VAL_train_Fea_Y_23',...
                'VAL_train_Fea_24','VAL_train_Fea_Y_24',...
                'VAL_train_Fea_34','VAL_train_Fea_Y_34',...
                'VAL_val_Fea_all','VAL_val_Fea_Y',...
                'TEST_train_Fea_1','TEST_train_Fea_Y_1',...
                'TEST_train_Fea_23','TEST_train_Fea_Y_23',...
                'TEST_train_Fea_24','TEST_train_Fea_Y_24',...
                'TEST_train_Fea_34','TEST_train_Fea_Y_34',...                
                'TEST_test_Fea_all','TEST_test_Fea_Y',...
                'SampleIndex_VAL_train',...
                'SampleIndex_VAL_val',...
                'SampleIndex_TEST_train',...
                'SampleIndex_TEST_test');

    end
    
    fprintf('\n\n');

end
                       
