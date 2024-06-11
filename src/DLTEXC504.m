%%  ����˵��
% ʵ�� 5.1-4
% ���ܣ�����AlexNet�����������������ͼ����з���
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��3��15��
% �汾��DLTEXC504-V1

%%  ����Ԥѵ���õ�AlexNet,��ȷ������������ͼ��Ĵ�С�Լ��������������
net = alexnet;                                   % ��Alexnet���빤����
inputSize = net.Layers(1).InputSize;             % ��ȡAlexnet�����������ͼ��Ĵ�С
classNames = net.Layers(end).ClassNames;         % ��ȡAlexnet������еķ���

%% ����MATLAB�Դ���RGBͼ�񣬸ı�ͼ���С���������
I = imread('peppers.png');
figure
imshow(I)
I = imresize(I,inputSize(1:2));
I = imnoise(I,'salt & pepper');   %��ӽ�������

%% ����Alexnet������������ͼ����з���
[label1,scores1] = classify(net,I);

%% ��ͼ������ʾ������������
figure
imshow(I)
title(string(label1) + ", " + num2str(100*scores1(classNames == label1),3) + "%");