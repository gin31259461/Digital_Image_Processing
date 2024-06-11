%%  ����˵��
% ʵ�� 4.1-1
% ���ܣ��Ժ���0~9���ֵĶ�ֵͼ������Ϊ28��28�����з��࣬���������׼ȷ��
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��2��22��
% �汾��DLTEX1-V1
% ע��1����ʵ����Ҫ����˵����ι������磿��θı�����ṹ������ṹ�ı���Ӱ�졣���ԣ����ص��ע����3
%     2����һЩ����ṹ�ĵ�����Ҫ�ı䲽��3�е���ز������ü���
%     3�����߿��Խ��ע�ͣ�%֮�����䣩�Գ���������
%
% �ڱ�������θı�����Ľṹ?
% 1)ȥ��������һ���㿴Ч�����ֱ��ڵ�һ��batchNormalizationLayer����41�У����ֱ��ڵڶ���batchNormalizationLayer����46�У�����Ӧ�����֮ǰ��%������ע�͵�
% 2)ȥ��һ������㿴Ч�����ֱ��ڵھ����2����45�У���������һ����2����46�У��������Լ����2����47�У����ػ���2����48�У�����Ӧ�����֮ǰ��%������ע�͵�
% 3)ȥ��һ�������֮�󣬼��پ���˵�������Ч������2���Ļ����ϣ��������1�ľ���˵ĸ�����Ϊ4����40�У������е�2������8��Ϊ4����

%% ����ڴ桢�����Ļ
clear
clc

%% ����1������ͼ���������ݣ�����ʾ���еĲ���ͼ�񣨱��ڲ��ص㽲�⣩
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

%% ����2�������ص�ͼ��������Ϊѵ�����Ͳ��Լ���ע���ڱ����У�ѵ����������Ϊ750����ʣ���Ϊ���Լ��������ڲ��ص㽲�⡿
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% ����3������������磨ע�������ڸò��ֽ�����ز��������øĽ����������ص㽲��Ĳ��֡�
layers = [
    imageInputLayer([28 28 1])                  % ����㣬1��ͨ��������Ϊ28��28
    
    convolution2dLayer([3 3],8,'Padding','same')% �����1������˴�СΪ3��3������˵ĸ���Ϊ8��ÿ������˵�ͨ����������ͼ���ͨ������ȣ�������ÿ�������1��ͨ��������ķ�ʽ��������䷽ʽ�����趨Ϊsame��ʽ��
    batchNormalizationLayer                     % ������һ����1
    reluLayer                                   % ReLu�����Լ����1
    maxPooling2dLayer(2,'Stride',2)             % �ػ���1���ػ���ʽ�����ػ����ػ�����Ϊ2��2������Ϊ2
    
   convolution2dLayer([3 3],16,'Padding','same')% �����2������˴�СΪ3��3������˵ĸ���Ϊ16��ÿ������˵�ͨ��������������ͼ��ͨ������ȣ�������ÿ�������8��ͨ��������ķ�ʽ��������䷽ʽ�����趨Ϊsame��ʽ��
   batchNormalizationLayer                      % ������һ����2
   reluLayer                                    % ReLu�����Լ����2
   maxPooling2dLayer(2,'Stride',2)              % �ػ���2���ػ���ʽ�����ػ����ػ�����Ϊ2��2������Ϊ2
    
    fullyConnectedLayer(10)                     % ȫ���Ӳ㣺��ȫ���Ӳ�����ĸ�������Ϊ10��
    softmaxLayer                                % softmaxLayer�㣺���ÿ������ĸ���
    classificationLayer ];                       % ����㣺������һ�������ĸ��ʣ����з��ಢ���

%% ����4������ѵ��ѡ���ʼѵ������ص�ѵ����������4.2���н�����ϸ���ܣ������ڲ��ص㽲�⡿
    options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');                % ����ѵ��ѡ��
                                                 %'sgdm'��ʾʹ�þ��ж���������ݶ��½�������ѵ�����磻'InitialLearnRate'���ó�ʼѧϰ��Ϊ0.01��'MaxEpochs'�����ѵ����������Ϊ4��'Shuffle'��ʾ�������ݣ�'every-epoch'�����ÿһ��ѵ��������һ�����ݣ�'ValidationData'����������֤���ݼ���'ValidationFrequency'������֤Ƶ��Ϊ30��'Verbose'����Ϊfalse����ʾ������Ϣ��'Plots'��ѵ������ͼ��

    net = trainNetwork(imdsTrain,layers,options); %���������ѵ��
    
   %% ����5����ѵ���õ��������ڶ��µ�����ͼ����з��࣬������׼ȷ�ʡ����ڲ��ص㽲�⡿
   YPred = classify(net,imdsValidation);
   YValidation = imdsValidation.Labels;
   accuracy = sum(YPred == YValidation)/numel(YValidation)
