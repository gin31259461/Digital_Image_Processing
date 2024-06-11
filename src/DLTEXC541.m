%%  程序说明
% 实例 5.4-1
% 功能：训练R-CNN用于识别交通标志
% 作者：zhaoxch_mail@sina.com
% 时间：2020年4月19日
% 版本：DLTEXC541-V1

%% 步骤1：构建一个卷积神经网络
% 输入层（与训练集图像的大小相同）
inputLayer = imageInputLayer([32 32 3]);
% 卷积层
filterSize = [5 5]; %卷积核大小
numFilters = 32; %卷积核个数
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
% 全连接层
finalLayers = [
fullyConnectedLayer(64)
batchNormalizationLayer
reluLayer
fullyConnectedLayer(10)
softmaxLayer
classificationLayer
];
% 构建整个卷积神经网络 
layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

%% 步骤2：采用CIFAR-10数据集，训练所构建的卷积神经网络；
% 导入CIFAR-10数据集，要求与步骤详见本书4.7节
[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load('cifar10Data');

% 显示其中的100幅
figure
thumbnails = trainingImages(:,:,:,1:100);
montage(thumbnails)

% 设置训练策略参数
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'MaxEpochs', 25, ...    
    'MiniBatchSize', 128, ...
    'Verbose', true);

% 训练网络，trainNetwork函数的参数依次分别为：训练数据集，训练集标签，网络结构，训练策略。
    cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);

%% 步骤3：验证卷积神经网络的分类效果.
YTest = classify(cifar10Net, testImages);
% 计算正确率.
accuracy = sum(YTest == testLabels)/numel(testLabels)

%% 步骤4：训练RCNN检测器
% 导入41张包括有“Stop sign”交通标志的图像
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;
% 设置图像路径参数
visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);
% 显示数据
summary(stopSignsAndCars)
% 只保留文件名及其所包含的“stop sign”区域
stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});
% 显示一张照片及其所包含的真实“stop sign”区域
I = imread(stopSigns.imageFilename{1});
I = insertObjectAnnotation(I,'Rectangle',stopSigns.stopSign{1},'stop sign','LineWidth',8);
figure
imshow(I)

% 设置训练策略
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 35, ...
        'Verbose', true);
    
  % 训练R-CNN网络.    
    rcnn = trainRCNNObjectDetector(stopSigns, cifar10Net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])

  % 载入测试图片
testImage = imread('stopSignTest.jpg');
%% 步骤5：检验检测器的对“Stop sign”交通标志的图像的检测效果
% 检测“Stop sign“标志
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128)
% 标注置信度
[score, idx] = max(score);
bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);
figure
imshow(outputImage)
