function csp_feature = Rsx_singlewindow_cspfeature(EEG_data,Csp_transpose,filter_num)
% ���������õõ���csp���󣬶����ݽ���ת����������ȡ 2021.01.16
%����:  EEg_data������ͨ������������
%       Csp_transpose: CSP��ȡ�ı任����
%       filter_num�� �˲����������ɶԳ��֣���ż��
%����� csp_feature��1* filter_num
%% ת������
    [m, n] = size(Csp_transpose);
    %check for valid dimensions
    if(m<filter_num)
        disp('Cannot reduce to a higher dimensional space!');
        return
    end
    %instantiate filter matrix
    Ptild = zeros(filter_num,n);
    %create the n-dimensional filter by sorting
    i=0;
    for d = 1:filter_num
        if(mod(d,2)==0)   
            Ptild(d,:) = Csp_transpose(m-i,:);
            i=i+1;
        else
            Ptild(d,:) = Csp_transpose(1+i,:);
        end
    end
    %get length of input signal
    T = size(EEG_data,2);
    %instantiate output matrix
    Trans_data = zeros(T, filter_num);
    %filtering
    for d = 1:filter_num
       for t = 1:T 
            Trans_data(t, d) = dot(Ptild(d,:),EEG_data(:,t));
       end
    end  
%% ��ȡ����
    LogDown = sum(var(Trans_data,1,1));
    LogUp = var(Trans_data,1,1);
    csp_feature = log(LogUp/LogDown);
end

