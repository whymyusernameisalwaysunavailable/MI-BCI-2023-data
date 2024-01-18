% Ver.:MdlInit_GenFeaSets0115(forsubmission)
% LastVer.:MdlInit_GenFeaSets1103
% LastVer.:MdlInit_GenFeaSets1019
%
% Model initialization of 6 binary models
%   Set parameters for data preprocessing
%   Extract data of 4 action types separately from raw data
%   generate feature sets of training and test samples
%
% 10 fold cross-validation
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
SlideWindowLength = 256/4;                                                 % Slide length
windows_per_trial = (data_points_per_TrainTrial - WindowLength) / SlideWindowLength + 1; % = 3
w_p_trl_select = 5;     
windows_per_action = windows_per_trial * trials_per_action;     
w_p_act_select = w_p_trl_select * trials_per_action;     

% assert((sample_frequency*secfromstart + WindowLength + (w_p_t_select-1)*SlideWindowLength)<=data_points_per_TrainTrial,'START POINT ERROR!')

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
Wband = [1,4;4,8;8,13;12,16;16,20;20,24;24,28;28,32;32,36;36,40;13,20;13,30];
number_bandpass_filters = size(Wband,1); 
FilterType = 'bandpass';
FilterOrder = 4;   

% choose CSP param
FilterNum = 2;% cannot be bigger than number of channels (?bandpass filters)

save('InitializationParameters.mat',...
              'sample_frequency',...
              'trials_per_action','seconds_per_TrainTrial',...
              'w_p_trl_select', 'w_p_act_select',...
              'WindowLength','SlideWindowLength',...
              'channels','Wband',...
              'FilterType','FilterOrder',...
              'FilterNum');

%% Load subject information for analysing
load('subjectinfo.mat','Subject');

for subjectnum = 1
    
    fprintf(['  被试',num2str(subjectnum),'：',Subject(subjectnum).name,'计时...\n']);
    fprintf(['  Subj.',num2str(subjectnum),': ',Subject(subjectnum).name,'ticing...\n']);tic;
    offlinepath = ['.\Subject_',num2str(subjectnum)];
    
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
    TrialDataUpdate_mark = Subject(subjectnum).pointmark;

    for secfromstart = 0:0.125:1 % 0:0.125:1
        fprintf(['      第 ',num2str(secfromstart/0.125 + 1),' 个起始点：\n'])
        fprintf(['      NO. ',num2str(secfromstart/0.125 + 1),' START POINT: \n'])
        mkdir([offlinepath,'\',num2str(secfromstart/0.125 + 1)]);
        
        % put all 2-sec windows of each action together for further update
        SlideSample_Idle = cell(1, w_p_act_select);                     % windows_per_action = 60;
        SlideSample_Walk = cell(1, w_p_act_select);
        SlideSample_Ascend = cell(1, w_p_act_select);
        SlideSample_Descend = cell(1, w_p_act_select);
        SlideSample_mark = cell(1, w_p_act_select);
        for i = 1:trials_per_action
            for j = 1:w_p_trl_select
                PointStart = (i-1)*data_points_per_TrainTrial + (j-1)*SlideWindowLength + secfromstart * sample_frequency;
                SlideSample_Idle{1, (i-1)*w_p_trl_select+j} = TrialDataUpdate_Idle(:,PointStart + 1:PointStart + WindowLength );
                SlideSample_Walk{1, (i-1)*w_p_trl_select+j} = TrialDataUpdate_Walk(:,PointStart + 1:PointStart + WindowLength );
                SlideSample_Ascend{1, (i-1)*w_p_trl_select+j} = TrialDataUpdate_Ascend(:,PointStart + 1:PointStart + WindowLength );
                SlideSample_Descend{1, (i-1)*w_p_trl_select+j} = TrialDataUpdate_Descend(:,PointStart + 1:PointStart + WindowLength );
                SlideSample_mark{1, (i-1)*w_p_trl_select+j} = TrialDataUpdate_mark(:,PointStart + 1:PointStart + WindowLength );
            end
        end
        startpoint = 5+secfromstart;
        endpoint = ((w_p_trl_select-1)*SlideWindowLength + WindowLength)/sample_frequency + secfromstart + 5;
%         save([offlinepath,'\',num2str(secfromstart/0.125 + 1),'\margin.mat'],'startpoint','endpoint');

        %% Get & save rejection logical vector
        for i = 1:w_p_act_select % if more than 1/4 sample contaminated, replace with []
            if sum(SlideSample_mark{i}(1,:)) > 512*1.1
                SlideSample_Idle{i} = [];
            end
            if sum(SlideSample_mark{i}(2,:)) > 512*1.1
                SlideSample_Walk{i} = [];
            end
            if sum(SlideSample_mark{i}(3,:)) > 512*1.1
                SlideSample_Ascend{i} = [];
            end
            if sum(SlideSample_mark{i}(4,:)) > 512*1.1
                SlideSample_Descend{i} = [];
            end
        end
        
        rejVec_Idle = cellfun(@isempty,SlideSample_Idle);
        rejVec_Walk = cellfun(@isempty,SlideSample_Walk);
        rejVec_Ascend = cellfun(@isempty,SlideSample_Ascend);
        rejVec_Descend = cellfun(@isempty,SlideSample_Descend);
        rejected_sample_Idle = find(rejVec_Idle==1);
        rejected_sample_Walk = find(rejVec_Walk==1);
        rejected_sample_Ascend = find(rejVec_Ascend==1);
        rejected_sample_Descend = find(rejVec_Descend==1);
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
        acceptVec_Idle = ~rejVec_Idle;                                          % 1*60 logical, "true(1)" for accepting
        acceptVec_Walk = ~rejVec_Walk;
        acceptVec_Ascend = ~rejVec_Ascend;
        acceptVec_Descend = ~rejVec_Descend;

        %% Get all filter samples
        fprintf('  划分子频带...\n');
        fprintf('  Subband filtering...\n');

        % shape: (number of bandpass filters, windows per action)
        FilterSample_Idle = cell(number_bandpass_filters, w_p_act_select);  % 4 Filtered Freq Bands * 60 Windows
        FilterSample_Walk = cell(number_bandpass_filters, w_p_act_select);
        FilterSample_Ascend = cell(number_bandpass_filters, w_p_act_select);
        FilterSample_Descend = cell(number_bandpass_filters, w_p_act_select);

        SampleIndex_all = zeros(4,w_p_act_select);                          % 4 rows -- 4 classes
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

       %% Make 10 training-testing pairs of sample sets, 
        foldnum = 10;                                                           % 1 trial per fold
        windows_per_fold = w_p_act_select/foldnum;                          % 20 * 3 / 20
        rng(1);
        cvIndices = crossvalind('Kfold',trials_per_action,foldnum);

        for valcount = 1:foldnum
            fprintf(['  Generating ',num2str(valcount),' feature sets...\n']);

            valindex_tmp = (cvIndices==valcount);
            valindex = repelem(valindex_tmp,w_p_trl_select);
            samplenum_Test = 4 * windows_per_fold;

            %% Generate training-test feature sets 
            %% Training set filter sample                          % _Train
            SampleIndex_Train = SampleIndex_all(:,~valindex);
            samplenum_Train = size(SampleIndex_Train,2);
            % actual indices of _Train samples, 0 for rejected samples

            % Get indices of replacing samples for rejected samples
            for classtype = 1:4
                for sampleIndex = 1:samplenum_Train
                    if SampleIndex_Train(classtype,sampleIndex) == 0
                        trialIndex = floor((sampleIndex-0.1)/5);
                        sampleIndex_1st = trialIndex * 5 + 1;
                        sampleIndex_tmp = SampleIndex_Train(classtype,sampleIndex_1st:sampleIndex_1st+4);
                        sampleIndex_replace = sampleIndex_tmp(sampleIndex_tmp ~= 0);
                        if ~isempty(sampleIndex_replace)
                            samplenum_replace = length(sampleIndex_replace);
                            SampleIndex_Train(classtype,sampleIndex) = sampleIndex_replace(mod(sampleIndex,samplenum_replace)+1);
                        else
                            trialIndex = abs(trialIndex - 1);
                            sampleIndex_1st = trialIndex * 5 + 1;
                            sampleIndex_tmp = SampleIndex_Train(classtype,sampleIndex_1st:sampleIndex_1st+4);
                            sampleIndex_replace = sampleIndex_tmp(sampleIndex_tmp ~= 0);
                            samplenum_replace = length(sampleIndex_replace);
                            SampleIndex_Train(classtype,sampleIndex) = sampleIndex_replace(mod(0,samplenum_replace)+1);
                            SampleIndex_Train(classtype,sampleIndex+1) = sampleIndex_replace(mod(1,samplenum_replace)+1);
                            SampleIndex_Train(classtype,sampleIndex+2) = sampleIndex_replace(mod(2,samplenum_replace)+1);
                            SampleIndex_Train(classtype,sampleIndex+3) = sampleIndex_replace(mod(3,samplenum_replace)+1);
                            SampleIndex_Train(classtype,sampleIndex+4) = sampleIndex_replace(mod(4,samplenum_replace)+1);
                        end
                    end
                end
            end

            FilterSample_Idle_Train = FilterSample_Idle(:,SampleIndex_Train(1,:));
            FilterSample_Walk_Train = FilterSample_Walk(:,SampleIndex_Train(2,:));
            FilterSample_Ascend_Train = FilterSample_Ascend(:,SampleIndex_Train(3,:));
            FilterSample_Descend_Train = FilterSample_Descend(:,SampleIndex_Train(4,:));

            %% Test set filter sample
            FilterSample_Test_init = [FilterSample_Idle(:,valindex), ...
                                         FilterSample_Walk(:,valindex), ...
                                         FilterSample_Ascend(:,valindex), ...
                                         FilterSample_Descend(:,valindex)];
            % [] for rejected sample
            FilterSample_Y_Test = [ones(windows_per_fold,1);ones(windows_per_fold,1)*2;...
                                      ones(windows_per_fold,1)*3;ones(windows_per_fold,1)*4];

            SampleIndex_Test_init = [SampleIndex_all(1,valindex),...
                                   SampleIndex_all(2,valindex),...
                                   SampleIndex_all(3,valindex),...
                                   SampleIndex_all(4,valindex)];
            % actual indices of _Test samples, 0 for rejected samples

            % Get indices of replacing samples for rejected samples
            % Fill replacing samples in []s for rejected samples
            FilterSample_Test = cell(size(FilterSample_Test_init));
            for sampleIndex = 1:samplenum_Test
                if SampleIndex_Test_init(sampleIndex) ~= 0
                    FilterSample_Test(:,sampleIndex) = FilterSample_Test_init(:,sampleIndex);
                    SampleIndex_Test(:,sampleIndex) = SampleIndex_Test_init(:,sampleIndex);
                else
                    trialIndex = floor((sampleIndex-0.1)/5);
                    sampleIndex_1st = trialIndex * 5 + 1;
                    sampleIndex_tmp = SampleIndex_Test_init(sampleIndex_1st:sampleIndex_1st+4);
                    sampleIndex_replace = sampleIndex_tmp(sampleIndex_tmp ~= 0);
                    FilterSample_test_tmp = FilterSample_Test_init(:,sampleIndex_1st:sampleIndex_1st+4);
                    FilterSample_test_replace = FilterSample_test_tmp(:,(sampleIndex_tmp ~= 0));
                    if ~isempty(sampleIndex_replace)
                        replacenum = mod(sampleIndex,length(sampleIndex_replace))+1;
                        SampleIndex_Test(sampleIndex) = sampleIndex_replace(replacenum);
                        FilterSample_Test(:,sampleIndex) = FilterSample_test_replace(:,replacenum);
                    else
                        SampleIndex_Test(sampleIndex) = 0;
                    end
                end
            end

            %% Geting CSP for TEST process
            fprintf('    生成csp矩阵...\n');
            fprintf('    Generating CSP matrices...\n');
            CspTranspose_12 = cell(1,number_bandpass_filters);
            CspTranspose_13 = cell(1,number_bandpass_filters);
            CspTranspose_14 = cell(1,number_bandpass_filters);
            CspTranspose_23 = cell(1,number_bandpass_filters);
            CspTranspose_24 = cell(1,number_bandpass_filters);
            CspTranspose_34 = cell(1,number_bandpass_filters);
            for i = 1:number_bandpass_filters
                CspTranspose_12{i} = Rsx_CSP_R3(FilterSample_Idle_Train(i,:),FilterSample_Walk_Train(i,:)); 
                CspTranspose_13{i} = Rsx_CSP_R3(FilterSample_Idle_Train(i,:),FilterSample_Ascend_Train(i,:)); 
                CspTranspose_14{i} = Rsx_CSP_R3(FilterSample_Idle_Train(i,:),FilterSample_Descend_Train(i,:)); 
                CspTranspose_23{i} = Rsx_CSP_R3(FilterSample_Walk_Train(i,:),FilterSample_Ascend_Train(i,:)); 
                CspTranspose_24{i} = Rsx_CSP_R3(FilterSample_Walk_Train(i,:),FilterSample_Descend_Train(i,:)); 
                CspTranspose_34{i} = Rsx_CSP_R3(FilterSample_Ascend_Train(i,:),FilterSample_Descend_Train(i,:));  
            end
            CSPs = cell(1,6);
            CSPs{1} = CspTranspose_12;
            CSPs{2} = CspTranspose_13;
            CSPs{3} = CspTranspose_14;
            CSPs{4} = CspTranspose_23;
            CSPs{5} = CspTranspose_24;
            CSPs{6} = CspTranspose_34;

            %% CSP Feature extraction for generating classifier (For TEST process)
            % CSP Feature extraction (of training samples)
            fprintf('    生成训练集特征及标签...\n');
            fprintf('    CSP Feature extraction for generating classifier...\n');
            % Idle vs. Walk
            % shape: (samplenum, number of bandpass filters * FilterNum)
            FeaSample_Idle_Train_12 = [];
            FeaSample_Walk_Train_12 = [];
            for i = 1:size(FilterSample_Idle_Train,2)                    %每个window的每个freq band进行特征提取
                FeaTemp_Idle = [];
                FeaTemp_Walk = [];
                for j =1:number_bandpass_filters
                    FeaTemp_Idle = [FeaTemp_Idle,Rsx_singlewindow_cspfeature(FilterSample_Idle_Train{j,i},CspTranspose_12{j},FilterNum)];
                    FeaTemp_Walk = [FeaTemp_Walk,Rsx_singlewindow_cspfeature(FilterSample_Walk_Train{j,i},CspTranspose_12{j},FilterNum)];
                end
                FeaSample_Idle_Train_12 = [FeaSample_Idle_Train_12;FeaTemp_Idle];
                FeaSample_Walk_Train_12 = [FeaSample_Walk_Train_12;FeaTemp_Walk];
            end

            % Idle vs. Ascend
            % shape: (samplenum, number of bandpass filters * FilterNum)
            FeaSample_Idle_Train_13 = [];
            FeaSample_Ascend_Train_13 = [];
            for i = 1:size(FilterSample_Idle_Train,2)                    %每个window的每个freq band进行特征提取
                FeaTemp_Idle = [];
                FeaTemp_Ascend = [];
                for j =1:number_bandpass_filters
                    FeaTemp_Idle = [FeaTemp_Idle,Rsx_singlewindow_cspfeature(FilterSample_Idle_Train{j,i},CspTranspose_13{j},FilterNum)];
                    FeaTemp_Ascend = [FeaTemp_Ascend,Rsx_singlewindow_cspfeature(FilterSample_Ascend_Train{j,i},CspTranspose_13{j},FilterNum)];
                end
                FeaSample_Idle_Train_13 = [FeaSample_Idle_Train_13;FeaTemp_Idle];
                FeaSample_Ascend_Train_13 = [FeaSample_Ascend_Train_13;FeaTemp_Ascend];
            end

            % Idle vs. Descend
            % shape: (samplenum, number of bandpass filters * FilterNum)
            FeaSample_Idle_Train_14 = [];
            FeaSample_Descend_Train_14 = [];
            for i = 1:size(FilterSample_Idle_Train,2)                    %每个window的每个freq band进行特征提取
                FeaTemp_Idle = [];
                FeaTemp_Descend = [];
                for j =1:number_bandpass_filters
                    FeaTemp_Idle = [FeaTemp_Idle,Rsx_singlewindow_cspfeature(FilterSample_Idle_Train{j,i},CspTranspose_14{j},FilterNum)];
                    FeaTemp_Descend = [FeaTemp_Descend,Rsx_singlewindow_cspfeature(FilterSample_Descend_Train{j,i},CspTranspose_14{j},FilterNum)];
                end
                FeaSample_Idle_Train_14 = [FeaSample_Idle_Train_14;FeaTemp_Idle];
                FeaSample_Descend_Train_14 = [FeaSample_Descend_Train_14;FeaTemp_Descend];
            end

            % Walk vs. Ascend
            % shape: (samplenum, number of bandpass filters * FilterNum)
            FeaSample_Walk_Train_23 = [];
            FeaSample_Ascend_Train_23 = [];
            for i = 1:size(FilterSample_Walk_Train,2)
                FeaTemp_Walk = [];
                FeaTemp_Ascend = [];
                for j =1:number_bandpass_filters
                    FeaTemp_Walk = [FeaTemp_Walk,Rsx_singlewindow_cspfeature(FilterSample_Walk_Train{j,i},CspTranspose_23{j},FilterNum)];
                    FeaTemp_Ascend = [FeaTemp_Ascend,Rsx_singlewindow_cspfeature(FilterSample_Ascend_Train{j,i},CspTranspose_23{j},FilterNum)];
                end
                FeaSample_Walk_Train_23 = [FeaSample_Walk_Train_23;FeaTemp_Walk];
                FeaSample_Ascend_Train_23 = [FeaSample_Ascend_Train_23;FeaTemp_Ascend];
            end

            % Walk vs. Descend
            % shape: (samplenum, number of bandpass filters * FilterNum)
            FeaSample_Walk_Train_24 = [];
            FeaSample_Descend_Train_24 = [];
            for i = 1:size(FilterSample_Walk_Train,2)
                FeaTemp_Walk = [];
                FeaTemp_Descend = [];
                for j =1:number_bandpass_filters
                    FeaTemp_Walk = [FeaTemp_Walk,Rsx_singlewindow_cspfeature(FilterSample_Walk_Train{j,i},CspTranspose_24{j},FilterNum)];
                    FeaTemp_Descend = [FeaTemp_Descend,Rsx_singlewindow_cspfeature(FilterSample_Descend_Train{j,i},CspTranspose_24{j},FilterNum)];
                end
                FeaSample_Walk_Train_24 = [FeaSample_Walk_Train_24;FeaTemp_Walk];
                FeaSample_Descend_Train_24 = [FeaSample_Descend_Train_24;FeaTemp_Descend];
            end

            % Ascend vs. Descend
            % shape: (samplenum, number of bandpass filters * FilterNum)
            FeaSample_Ascend_Train_34 = [];
            FeaSample_Descend_Train_34 = [];
            for i = 1:size(FilterSample_Walk_Train,2)
                FeaTemp_Ascend = [];
                FeaTemp_Descend = [];
                for j =1:number_bandpass_filters
                    FeaTemp_Ascend = [FeaTemp_Ascend,Rsx_singlewindow_cspfeature(FilterSample_Ascend_Train{j,i},CspTranspose_34{j},FilterNum)];
                    FeaTemp_Descend = [FeaTemp_Descend,Rsx_singlewindow_cspfeature(FilterSample_Descend_Train{j,i},CspTranspose_34{j},FilterNum)];
                end
                FeaSample_Ascend_Train_34 = [FeaSample_Ascend_Train_34;FeaTemp_Ascend];
                FeaSample_Descend_Train_34 = [FeaSample_Descend_Train_34;FeaTemp_Descend];
            end

            %% Min-max normalization
            % Idle vs. Walk
            Train_Fea_12_init = [FeaSample_Idle_Train_12;FeaSample_Walk_Train_12]; 
            Train_Fea_Y_12 = [ones(size(FeaSample_Idle_Train_12,1),1);ones(size(FeaSample_Walk_Train_12,1),1)*2];
            FeaMin_12 = min(Train_Fea_12_init);
            FeaMax_12 = max(Train_Fea_12_init);
            Train_Fea_12 = (Train_Fea_12_init - FeaMin_12)./(FeaMax_12-FeaMin_12);

            % Idle vs. Ascend
            Train_Fea_13_init = [FeaSample_Idle_Train_13;FeaSample_Ascend_Train_13]; 
            Train_Fea_Y_13 = [ones(size(FeaSample_Idle_Train_13,1),1);ones(size(FeaSample_Ascend_Train_13,1),1)*2];
            FeaMin_13 = min(Train_Fea_13_init);
            FeaMax_13 = max(Train_Fea_13_init);
            Train_Fea_13 = (Train_Fea_13_init - FeaMin_13)./(FeaMax_13-FeaMin_13);

            % Idle vs. Descend
            Train_Fea_14_init = [FeaSample_Idle_Train_14;FeaSample_Descend_Train_14]; 
            Train_Fea_Y_14 = [ones(size(FeaSample_Idle_Train_14,1),1);ones(size(FeaSample_Descend_Train_14,1),1)*2];
            FeaMin_14 = min(Train_Fea_14_init);
            FeaMax_14 = max(Train_Fea_14_init);
            Train_Fea_14 = (Train_Fea_14_init - FeaMin_14)./(FeaMax_14-FeaMin_14);

            % Walk vs. Ascend
            Train_Fea_23_init = [FeaSample_Walk_Train_23;FeaSample_Ascend_Train_23];
            Train_Fea_Y_23 = [ones(size(FeaSample_Walk_Train_23,1),1);ones(size(FeaSample_Ascend_Train_23,1),1)*2];
            FeaMin_23 = min(Train_Fea_23_init);
            FeaMax_23 = max(Train_Fea_23_init);
            Train_Fea_23 = (Train_Fea_23_init - FeaMin_23)./(FeaMax_23-FeaMin_23);

            % Walk vs. Descend
            Train_Fea_24_init = [FeaSample_Walk_Train_24;FeaSample_Descend_Train_24];
            Train_Fea_Y_24 = [ones(size(FeaSample_Walk_Train_24,1),1);ones(size(FeaSample_Descend_Train_24,1),1)*2];
            FeaMin_24 = min(Train_Fea_24_init);
            FeaMax_24 = max(Train_Fea_24_init);
            Train_Fea_24 = (Train_Fea_24_init - FeaMin_24)./(FeaMax_24-FeaMin_24);

            % Ascend vs. Descend
            Train_Fea_34_init = [FeaSample_Ascend_Train_34;FeaSample_Descend_Train_34];
            Train_Fea_Y_34 = [ones(size(FeaSample_Ascend_Train_34,1),1);ones(size(FeaSample_Descend_Train_34,1),1)*2];
            FeaMin_34 = min(Train_Fea_34_init);
            FeaMax_34 = max(Train_Fea_34_init);
            Train_Fea_34 = (Train_Fea_34_init - FeaMin_34)./(FeaMax_34-FeaMin_34);

            %% CSP Feature extraction for classifier test 
            % CSP Feature extraction (of test samples)
            fprintf('    生成测试集特征及标签...\n');
            fprintf('    CSP Feature extraction for classifier test ...\n');
            samplenum_Test = sum(SampleIndex_Test ~= 0);                  % in case a whole trial is rejected
            acceptIndex_Test = find(SampleIndex_Test ~= 0);
            Test_Fea_12 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
            Test_Fea_13 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
            Test_Fea_14 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
            Test_Fea_23 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
            Test_Fea_24 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
            Test_Fea_34 = zeros(windows_per_fold * 4,number_bandpass_filters * FilterNum);
            for i = 1:samplenum_Test                                         %每个window的每个freq band进行特征提取
                FeaTemp_12 = [];
                FeaTemp_13 = [];
                FeaTemp_14 = [];
                FeaTemp_23 = [];
                FeaTemp_24 = [];
                FeaTemp_34 = [];
                for j =1:number_bandpass_filters
                    FeaTemp_12 = [FeaTemp_12,Rsx_singlewindow_cspfeature(FilterSample_Test{j,acceptIndex_Test(i)},CspTranspose_12{j},FilterNum)];
                    FeaTemp_13 = [FeaTemp_13,Rsx_singlewindow_cspfeature(FilterSample_Test{j,acceptIndex_Test(i)},CspTranspose_13{j},FilterNum)];
                    FeaTemp_14 = [FeaTemp_14,Rsx_singlewindow_cspfeature(FilterSample_Test{j,acceptIndex_Test(i)},CspTranspose_14{j},FilterNum)];
                    FeaTemp_23 = [FeaTemp_23,Rsx_singlewindow_cspfeature(FilterSample_Test{j,acceptIndex_Test(i)},CspTranspose_23{j},FilterNum)];
                    FeaTemp_24 = [FeaTemp_24,Rsx_singlewindow_cspfeature(FilterSample_Test{j,acceptIndex_Test(i)},CspTranspose_24{j},FilterNum)];
                    FeaTemp_34 = [FeaTemp_34,Rsx_singlewindow_cspfeature(FilterSample_Test{j,acceptIndex_Test(i)},CspTranspose_34{j},FilterNum)];
                end
                Test_Fea_12(acceptIndex_Test(i),:) = (FeaTemp_12 - FeaMin_12)./(FeaMax_12 - FeaMin_12);
                Test_Fea_13(acceptIndex_Test(i),:) = (FeaTemp_13 - FeaMin_13)./(FeaMax_13 - FeaMin_13);
                Test_Fea_14(acceptIndex_Test(i),:) = (FeaTemp_14 - FeaMin_14)./(FeaMax_14 - FeaMin_14);
                Test_Fea_23(acceptIndex_Test(i),:) = (FeaTemp_23 - FeaMin_23)./(FeaMax_23 - FeaMin_23);
                Test_Fea_24(acceptIndex_Test(i),:) = (FeaTemp_24 - FeaMin_24)./(FeaMax_24 - FeaMin_24);
                Test_Fea_34(acceptIndex_Test(i),:) = (FeaTemp_34 - FeaMin_34)./(FeaMax_34 - FeaMin_34);
            end

            Test_Fea_all = cell(1,6);
            Test_Fea_all{1} = Test_Fea_12;
            Test_Fea_all{2} = Test_Fea_13;
            Test_Fea_all{3} = Test_Fea_14;
            Test_Fea_all{4} = Test_Fea_23;
            Test_Fea_all{5} = Test_Fea_24;
            Test_Fea_all{6} = Test_Fea_34;
            Test_Fea_Y = FilterSample_Y_Test;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% CSP Feature extraction for classifier test 
            % CSP Feature extraction (of test samples)
            fprintf('    训练集特征检查...\n');
            FilterSample_Train = [FilterSample_Idle_Train,FilterSample_Walk_Train,FilterSample_Ascend_Train,FilterSample_Descend_Train];
            trx_Fea_12 = zeros(samplenum_Train * 4,number_bandpass_filters * FilterNum);
            trx_Fea_13 = zeros(samplenum_Train * 4,number_bandpass_filters * FilterNum);
            trx_Fea_14 = zeros(samplenum_Train * 4,number_bandpass_filters * FilterNum);
            trx_Fea_23 = zeros(samplenum_Train * 4,number_bandpass_filters * FilterNum);
            trx_Fea_24 = zeros(samplenum_Train * 4,number_bandpass_filters * FilterNum);
            trx_Fea_34 = zeros(samplenum_Train * 4,number_bandpass_filters * FilterNum);
            for i = 1:samplenum_Train*4                                         %每个window的每个freq band进行特征提取
                FeaTemp_12 = [];
                FeaTemp_13 = [];
                FeaTemp_14 = [];
                FeaTemp_23 = [];
                FeaTemp_24 = [];
                FeaTemp_34 = [];
                for j =1:number_bandpass_filters
                    FeaTemp_12 = [FeaTemp_12,Rsx_singlewindow_cspfeature(FilterSample_Train{j,i},CspTranspose_12{j},FilterNum)];
                    FeaTemp_13 = [FeaTemp_13,Rsx_singlewindow_cspfeature(FilterSample_Train{j,i},CspTranspose_13{j},FilterNum)];
                    FeaTemp_14 = [FeaTemp_14,Rsx_singlewindow_cspfeature(FilterSample_Train{j,i},CspTranspose_14{j},FilterNum)];
                    FeaTemp_23 = [FeaTemp_23,Rsx_singlewindow_cspfeature(FilterSample_Train{j,i},CspTranspose_23{j},FilterNum)];
                    FeaTemp_24 = [FeaTemp_24,Rsx_singlewindow_cspfeature(FilterSample_Train{j,i},CspTranspose_24{j},FilterNum)];
                    FeaTemp_34 = [FeaTemp_34,Rsx_singlewindow_cspfeature(FilterSample_Train{j,i},CspTranspose_34{j},FilterNum)];
                end
                trx_Fea_12(i,:) = (FeaTemp_12 - FeaMin_12)./(FeaMax_12 - FeaMin_12);
                trx_Fea_13(i,:) = (FeaTemp_13 - FeaMin_13)./(FeaMax_13 - FeaMin_13);
                trx_Fea_14(i,:) = (FeaTemp_14 - FeaMin_14)./(FeaMax_14 - FeaMin_14);
                trx_Fea_23(i,:) = (FeaTemp_23 - FeaMin_23)./(FeaMax_23 - FeaMin_23);
                trx_Fea_24(i,:) = (FeaTemp_24 - FeaMin_24)./(FeaMax_24 - FeaMin_24);
                trx_Fea_34(i,:) = (FeaTemp_34 - FeaMin_34)./(FeaMax_34 - FeaMin_34);
            end

            trx_Fea_all = cell(1,6);
            trx_Fea_all{1} = trx_Fea_12;
            trx_Fea_all{2} = trx_Fea_13;
            trx_Fea_all{3} = trx_Fea_14;
            trx_Fea_all{4} = trx_Fea_23;
            trx_Fea_all{5} = trx_Fea_24;
            trx_Fea_all{6} = trx_Fea_34;
            trx_Fea_Y = [ones(samplenum_Train,1);ones(samplenum_Train,1)*2;ones(samplenum_Train,1)*3;ones(samplenum_Train,1)*4];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            toc;
            %% Save feature sets
            save([offlinepath,'\',num2str(secfromstart/0.125 + 1),...
                '\FeaSets_',num2str(valcount),'_for_classifiers_trte.mat'],...
                    'Train_Fea_12','Train_Fea_Y_12',...
                    'Train_Fea_13','Train_Fea_Y_13',...
                    'Train_Fea_14','Train_Fea_Y_14',...
                    'Train_Fea_23','Train_Fea_Y_23',...
                    'Train_Fea_24','Train_Fea_Y_24',...
                    'Train_Fea_34','Train_Fea_Y_34',...
                    'Test_Fea_all','Test_Fea_Y',...
                    'CSPs',...
                    'trx_Fea_all','trx_Fea_Y',...
                    'SampleIndex_Train',...
                    'SampleIndex_Test');

        end
        fprintf(['    第',num2str(secfromstart/0.125 + 1),'次起始点特征提取完毕！！！\n']);toc;
        fprintf(['    NO.',num2str(secfromstart/0.125 + 1),' START POINT FEA-EXTRACTION FINISHED！！！\n']);toc;
    end
    fprintf('\n\n');

end
                       
