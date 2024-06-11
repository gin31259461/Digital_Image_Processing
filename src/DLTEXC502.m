
%%  ����˵��
% ʵ�� 5.1-1
% ���ܣ�����GoogleNet����������MATLAB�Դ�ͼ����з���
% ���ߣ�zhaoxch_mail@sina.com
% ʱ�䣺2020��3��1��
% �汾��DLTEXC501-V1

%%  ����Ԥѵ���õ�GoogleNet,��ȷ������������ͼ��Ĵ�С�Լ��������������
net = googlenet;                                   % ��GoogleNet���빤����
inputSize = net.Layers(1).InputSize;             % ��ȡGoogleNet�����������ͼ��Ĵ�С
classNames = net.Layers(end).ClassNames;         % ��ȡGoogleNet������еķ���

%% ��������RGBͼ�񣬲���ͼ��Ĵ�С�任����GoogleNet�����������ͼ����ͬ�Ĵ�С
I = imread('deer.jpg');
figure
imshow(I)
I = imresize(I,inputSize(1:2));

J= imread('horse.jpg');
figure
imshow(J)
J = imresize(J,inputSize(1:2));

%% ����GoogleNet�����������ͼ����з���
[label1,scores1] = classify(net,I);
[label2,scores2] = classify(net,J);

%% ��ͼ������ʾ������������
figure
imshow(I)
title(string(label1) + ", " + num2str(100*scores1(classNames == label1),3) + "%");

figure
imshow(J)
title(string(label2) + ", " + num2str(100*scores1(classNames == label2),3) + "%");



