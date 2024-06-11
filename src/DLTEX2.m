%%  ����˵��
% ʵ�� 4.2-1
% ���ܣ�������ͼ�������ֵ���б�ǶȽ���Ԥ�⣬����Ԥ��׼ȷ�ʺ;��������(RMSE)
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��2��29��
% �汾��DLTEX2-V1
% ע��1����ʵ����Ҫ����˵����ζԾ���������ѵ�����ı�ѵ���������Ӱ�졣���ԣ����ص��ע����3������4
%     2��Ҫ�ı�ѵ����������Ҫ�ڲ���3����ѵ��ѡ�����޸�
%     3�����߿��Խ��ע�ͣ�%֮�����䣩�Գ���������
%
% ���ʵ��1���ı�����ѵ��ѡ���еĲ�������ѵ��Ч����Σ�
%            1) ����ʼѧϰ�ʸ�Ϊ0.01����Ч����Σ�����������������69�е�0.001��Ϊ0.01����
%            2) ����ADAM��ѵ����������Ч����Σ����������������е�66��sgdm��Ϊadam��)
%            3��ȥ��Dropout�㣬��Ч����Σ����������������е�59�� dropoutLayer(0.2)��֮ǰ��%������ע�͵�)
% ���ʵ��2���ڼ���һ������㣬��Ԥ��Ч����Σ�
%            ������������55-57��֮ǰ��%ȥ������

%% ����ڴ桢�����Ļ
clear
clc

%% ����1�����غ���ʾͼ������
[XTrain,~,YTrain] = digitTrain4DArrayData;                       %����ѵ��ͼ������
[XValidation,~,YValidation] = digitTest4DArrayData;              %������֤ͼ������

% �����ʾ20��ѵ��ͼ��
numTrainImages = numel(YTrain);                                  %ͳ������ѵ������������
figure
idx = randperm(numTrainImages,20);
for i = 1:numel(idx)
    subplot(4,5,i)    
    imshow(XTrain(:,:,:,idx(i)))
    drawnow
end

%% ����2:�������������

layers = [
    imageInputLayer([28 28 1])                                 % ����㣬1��ͨ��������Ϊ28��28
 
    convolution2dLayer(3,8,'Padding','same')                   % �����1������˴�СΪ3��3������˵ĸ���Ϊ8��ÿ������˵�ͨ����������ͼ���ͨ������ȣ�������ÿ�������1��ͨ��������ķ�ʽ��������䷽ʽ�����趨Ϊsame��ʽ��
    batchNormalizationLayer                                    % ������һ����1
    reluLayer                                                  % ReLU�����Լ����1
    averagePooling2dLayer(2,'Stride',2)                        % �ػ���1���ػ���ʽ��ƽ���ػ����ػ�����Ϊ2��2������Ϊ2
 
    convolution2dLayer(3,16,'Padding','same')                  % �����2������˴�СΪ3��3������˵ĸ���Ϊ16��ÿ������˵�ͨ��������������ͼ��ͨ������ȣ�������ÿ�������8��ͨ��������ķ�ʽ��������䷽ʽ�����趨Ϊsame��ʽ��
    batchNormalizationLayer                                    % ������һ����2    
    reluLayer                                                  % ReLU�����Լ����2
    averagePooling2dLayer(2,'Stride',2)                        % �ػ���2���ػ���ʽ��ƽ���ػ����ػ�����Ϊ2��2������Ϊ2
  
    convolution2dLayer(3,32,'Padding','same')                  % �����3������˴�СΪ3��3������˵ĸ���Ϊ32��ÿ������˵�ͨ��������������ͼ��ͨ������ȣ�������ÿ�������16��ͨ��������ķ�ʽ��������䷽ʽ�����趨Ϊsame��ʽ��
    batchNormalizationLayer                                    % ������һ����3 
    reluLayer                                                  % ReLU�����Լ����3
    
    % convolution2dLayer(3,64,'Padding','same')
    % batchNormalizationLayer
    % reluLayer
    
    dropoutLayer(0.2)                                         % dropout�㣬�����20%����������
    fullyConnectedLayer(1)                                    % ȫ���Ӳ�,ȫ���Ӳ�����Ϊ1
    regressionLayer ];                                        % �ع�㣬����Ԥ����

%% ����3������ѵ��ѡ��
miniBatchSize  = 128;                                         % ѵ��һ����С��������Ϊ128
validationFrequency = floor(numel(YTrain)/miniBatchSize);     % ��֤Ƶ��
options = trainingOptions('sgdm', ...                         % ����ѵ�������������н�������ΪSGDM��
    'MiniBatchSize',miniBatchSize, ...                        % ������С����ѵ�������������н�������Ϊ128
    'MaxEpochs',30, ...                                       % �������ѵ���������ڱ������У����ѵ������Ϊ30
    'InitialLearnRate',0.001, ...                             % ���ó�ʼѧϰ��Ϊ0.001
    'LearnRateSchedule','piecewise', ...                      % ���ó�ʼ��ѧϰ���Ǳ仯��
    'LearnRateDropFactor',0.1, ...                            % ����ѧϰ��˥������Ϊ0.1
    'LearnRateDropPeriod',20, ...                             % ����ѧϰ��˥������Ϊ20�֣�����ÿ20�֣���֮ǰ��ѧϰ�ʻ����ϣ�����ѧϰ�ʵ�˥������0.1
    'Shuffle','every-epoch', ...                              % ����ÿһ�ֶ���������
    'ValidationData',{XValidation,YValidation}, ...           % ������֤�õ�����
    'ValidationFrequency',validationFrequency, ...            % ������֤Ƶ��
    'Plots','training-progress', ...                          % ���ô�ѵ������ͼ
    'Verbose',true);                                         % ���ùر�����ڵ����


%% ����4��ѵ������
net = trainNetwork(XTrain,YTrain,layers,options);

%% ����5������������
YPredicted = predict(net,XValidation);                       % ��ѵ���õ�����Ԥ����֤ͼ����������б�ĽǶ�
predictionError = YValidation - YPredicted;                  % ����Ԥ����б�ǶȺ�ʵ����б�Ƕ�֮���Ԥ�����
% ����׼ȷ��
thr = 10;                                                    % �趨��ֵ���ڱ����У���ֵ�趨Ϊ10��
numCorrect = sum(abs(predictionError) < thr);                % ��Ԥ��ֵ��ʵ��ֵ�����С��10��ʱ������ΪԤ����ȷ
numValidationImages = numel(YValidation);                    % ������֤ͼ�������
Accuracy = numCorrect/numValidationImages                    % ����׼ȷ��             
% ����RMSE��ֵ
squares = predictionError.^2;
RMSE = sqrt(mean(squares))

