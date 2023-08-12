% Enter unique information of subjects
% Information includes:
%       	 number
%       	 name
%       	 deleted samples (contaminated by artifacts)

fclose all;clc;clear;close all;

% all_people = 1:22;
% num_all = length(all_people);
% selected = [2,4:6,8:15,16:17,19,20:22];
selected = [1:14];
num_selected = length(selected);
% healthy_all = [7:15,19];
% num_allhealthy = length(healthy_all);
% healthy_selected = [8:15,19];
% num_selectedhealthy = length(healthy_selected);
% patient_all = [1:6,16:18,20:22];
% num_allpatient = length(patient_all);
% patient_selected = [2,4:6,16:17,20:22];
% num_selectedpatient = length(patient_selected);

for i = 1:num_selected
    Subject(i).number = selected(i);
    files = dir(['.\Subject_',num2str(selected(i)),'\*.txt']);
    Subject(i).name = files.name(1:end-4);
    if files.name(1:7) == 'patient'
        Subject(i).ishealthy = 0;
    else
        Subject(i).ishealthy = 1;
    end
end

Subject(1).rejected_sample_Idle = [7, 8, 16, 17, 18, 25, 26, 34, 35, 36, 51, 58];
Subject(1).rejected_sample_Walk = [31, 32, 33];
Subject(1).rejected_sample_Ascend = [];
Subject(1).rejected_sample_Descend = [36];

Subject(2).rejected_sample_Idle = [];
Subject(2).rejected_sample_Walk = [];
Subject(2).rejected_sample_Ascend = [];
Subject(2).rejected_sample_Descend = [];

Subject(3).rejected_sample_Idle = [];
Subject(3).rejected_sample_Walk = [10, 11];
Subject(3).rejected_sample_Ascend = [];
Subject(3).rejected_sample_Descend = [];

Subject(4).rejected_sample_Idle = [];
Subject(4).rejected_sample_Walk = [36, 53, 54];
Subject(4).rejected_sample_Ascend = [56, 57];
Subject(4).rejected_sample_Descend = [];

Subject(5).rejected_sample_Idle = [22, 23, 29, 30, 42];
Subject(5).rejected_sample_Walk = [34, 35, 36];
Subject(5).rejected_sample_Ascend = [37];
Subject(5).rejected_sample_Descend = [43];

Subject(6).rejected_sample_Idle = [56, 57];
Subject(6).rejected_sample_Walk = [];
Subject(6).rejected_sample_Ascend = [43, 44, 53, 54, 56, 57];
Subject(6).rejected_sample_Descend = [43];

Subject(7).rejected_sample_Idle = [];
Subject(7).rejected_sample_Walk = [];
Subject(7).rejected_sample_Ascend = [];
Subject(7).rejected_sample_Descend = [];

Subject(8).rejected_sample_Idle = [34, 35];
Subject(8).rejected_sample_Walk = [43, 44];
Subject(8).rejected_sample_Ascend = [13, 14, 15];
Subject(8).rejected_sample_Descend = [];

Subject(9).rejected_sample_Idle = [];
Subject(9).rejected_sample_Walk = [];
Subject(9).rejected_sample_Ascend = [49, 55];
Subject(9).rejected_sample_Descend = [];

Subject(10).rejected_sample_Idle = [1, 19, 22, 35, 36, 52, 53];
Subject(10).rejected_sample_Walk = [25, 26, 43, 44, 56, 57];
Subject(10).rejected_sample_Ascend = [16, 17, 21, 49];
Subject(10).rejected_sample_Descend = [12, 22, 23, 32, 33, 34, 44, 45, 49];

Subject(11).rejected_sample_Idle = [57];
Subject(11).rejected_sample_Walk = [];
Subject(11).rejected_sample_Ascend = [];
Subject(11).rejected_sample_Descend = [];

Subject(12).rejected_sample_Idle = [];
Subject(12).rejected_sample_Walk = [];
Subject(12).rejected_sample_Ascend = [];
Subject(12).rejected_sample_Descend = [];

Subject(13).rejected_sample_Idle = [];
Subject(13).rejected_sample_Walk = [];
Subject(13).rejected_sample_Ascend = [];
Subject(13).rejected_sample_Descend = [];

Subject(14).rejected_sample_Idle = [];
Subject(14).rejected_sample_Walk = [];
Subject(14).rejected_sample_Ascend = [];
Subject(14).rejected_sample_Descend = [];
save('subjectinfo.mat','Subject');

