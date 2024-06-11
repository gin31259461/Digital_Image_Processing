%%  ����˵��
% ʵ�� 5.4-1
% ���ܣ�ѵ��R-CNN����ʶ��ͨ��־
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��4��19��
% �汾��DLTEXC541-V1

%% ����1������һ�����������
% ����㣨��ѵ����ͼ��Ĵ�С��ͬ��
inputLayer = imageInputLayer([32 32 3]);
% �����
filterSize = [5 5]; %����˴�С
numFilters = 32; %����˸���
middleLayers = [
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(3, 'Stride', 2)
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(3, 'Stride',2)
convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(3, 'Stride',2)
];
% ȫ���Ӳ�
finalLayers = [
fullyConnectedLayer(64)
batchNormalizationLayer
reluLayer
fullyConnectedLayer(10)
softmaxLayer
classificationLayer
];
% ����������������� 
layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

%% ����2������CIFAR-10���ݼ���ѵ���������ľ�������磻
% ����CIFAR-10���ݼ���Ҫ���벽���������4.7��
[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load('cifar10Data');

% ��ʾ���е�100��
figure
thumbnails = trainingImages(:,:,:,1:100);
montage(thumbnails)

% ����ѵ�����Բ���
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'MaxEpochs', 25, ...    
    'MiniBatchSize', 128, ...
    'Verbose', true);

% ѵ�����磬trainNetwork�����Ĳ������ηֱ�Ϊ��ѵ�����ݼ���ѵ������ǩ������ṹ��ѵ�����ԡ�
    cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);

%% ����3����֤���������ķ���Ч��.
YTest = classify(cifar10Net, testImages);
% ������ȷ��.
accuracy = sum(YTest == testLabels)/numel(testLabels)

%% ����4��ѵ��RCNN�����
% ����41�Ű����С�Stop sign����ͨ��־��ͼ��
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;
% ����ͼ��·������
visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);
% ��ʾ����
summary(stopSignsAndCars)
% ֻ�����ļ��������������ġ�stop sign������
stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});
% ��ʾһ����Ƭ��������������ʵ��stop sign������
I = imread(stopSigns.imageFilename{1});
I = insertObjectAnnotation(I,'Rectangle',stopSigns.stopSign{1},'stop sign','LineWidth',8);
figure
imshow(I)

% ����ѵ������
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 35, ...
        'Verbose', true);
    
  % ѵ��R-CNN����.    
    rcnn = trainRCNNObjectDetector(stopSigns, cifar10Net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])

  % �������ͼƬ
testImage = imread('stopSignTest.jpg');
%% ����5�����������Ķԡ�Stop sign����ͨ��־��ͼ��ļ��Ч��
% ��⡰Stop sign����־
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128)
% ��ע���Ŷ�
[score, idx] = max(score);
bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);
figure
imshow(outputImage)
