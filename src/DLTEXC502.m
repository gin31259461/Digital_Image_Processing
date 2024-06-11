
%%  程序说明
% 实例 5.1-1
% 功能：基于GoogleNet卷积神经网络对MATLAB自带图像进行分类
% 作者：zhaoxch_mail@sina.com
% 时间：2020年3月1日
% 版本：DLTEXC501-V1

%%  导入预训练好的GoogleNet,并确定该网络输入图像的大小以及分类种类的名称
net = googlenet;                                   % 将GoogleNet导入工作区
inputSize = net.Layers(1).InputSize;             % 获取GoogleNet输入层中输入图像的大小
classNames = net.Layers(end).ClassNames;         % 获取GoogleNet输出层中的分类

%% 读入两幅RGB图像，并将图像的大小变换成与GoogleNet输入层中输入图像相同的大小
I = imread('deer.jpg');
figure
imshow(I)
I = imresize(I,inputSize(1:2));

J= imread('horse.jpg');
figure
imshow(J)
J = imresize(J,inputSize(1:2));

%% 基于GoogleNet对两幅输入的图像进行分类
[label1,scores1] = classify(net,I);
[label2,scores2] = classify(net,J);

%% 在图像上显示分类结果及概率
figure
imshow(I)
title(string(label1) + ", " + num2str(100*scores1(classNames == label1),3) + "%");

figure
imshow(J)
title(string(label2) + ", " + num2str(100*scores1(classNames == label2),3) + "%");



