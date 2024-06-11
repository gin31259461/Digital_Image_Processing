%%  程序说明
% 实例 5.1-4
% 功能：基于AlexNet卷积神经网络对添加噪声图像进行分类
% 作者：zhaoxch_mail@sina.com
% 时间：2020年3月15日
% 版本：DLTEXC504-V1

%%  导入预训练好的AlexNet,并确定该网络输入图像的大小以及分类种类的名称
net = alexnet;                                   % 将Alexnet导入工作区
inputSize = net.Layers(1).InputSize;             % 获取Alexnet输入层中输入图像的大小
classNames = net.Layers(end).ClassNames;         % 获取Alexnet输出层中的分类

%% 读入MATLAB自带的RGB图像，改变图像大小并添加噪声
I = imread('peppers.png');
figure
imshow(I)
I = imresize(I,inputSize(1:2));
I = imnoise(I,'salt & pepper');   %添加椒盐噪声

%% 基于Alexnet对添加噪声后的图像进行分类
[label1,scores1] = classify(net,I);

%% 在图像上显示分类结果及概率
figure
imshow(I)
title(string(label1) + ", " + num2str(100*scores1(classNames == label1),3) + "%");