clc
clear all
close all

%% 加载遥感数据
load UPavia.mat   % 数据分为四部分，总数据：Pavia，RGB显示：RGB_Pavia，用于训练：Train_Pavia,用于测试：Test_Pavia
% 说明：Train_Pavia每类样本数据固定为50，不便于后续样本数量调整，故选择Test_Pavia，将其分类训练和测试数据集，此样例程序中使用每类样本数100

Data=double(Pavia);
grd = Test_Pavia; %测试数据集
% clear Pavia

X=reshape(Data,size(Data,1)*size(Data,2),size(Data,3)); %总的原始数据，变为行*列的形式

%% 统计类别数
numlabeled=grd(grd>0); % 测试数据集的非背景数据
numN=unique(numlabeled); % 类别编号，共9类
num=length(numN); % 类别数量
T=[];
N=[];
for i=1:num
    N=length(find(grd ==i));
    T=[T,N]; % 测试数据集中每类别的像元数量
end

class_num=num;% 类别数量

%% show  the ground truth of hyperspectral data set
%colormap(grd);
figure,image(RGB_Pavia) ; %显示RGB影像

%% 选择训练样本的数量
numsample=100; % 每类样本选择100个，此处可以进行其他数量设置
[trainlabels,testlabels]=getlabeled(grd, numsample); % 获取训练和测试标签
%trainlabels = Train_Pavia;

disp('总体训练样本的数量为:')
Num_train=length(find(trainlabels~=0)) % 目前参数总共选择样本为：100*9=900

[xxt,yyt]=find(trainlabels~=0);
for i=1:size(xxt,1)
   Xtrain(i,:)=Data(xxt(i),yyt(i),:); % Data总的原始数据
   train_label(i)=trainlabels(xxt(i),yyt(i));
end

options = [];
options.ReducedDim = 15; %PCA降维为15维，此为影像特征
M = PCA(Xtrain, options);
Xtr=Xtrain*M; % 降维后的训练数据特征
Xts=X*M; % 降维后的总数据特征

% *** 分类器使用说明：用到哪个分类器，就把哪个分类器的注释去掉，程序就可运行 ***
% ==========(1)KNN分类器==============
k = 1;
class_results = knn(k, Xtr', train_label, Xts'); %Xts输入为总数据，class_results输出为总的预测输出结果
    if k == 1
        testResults = class_results; % 命名有歧义，容易误导
    else
        [maxCount,idx] = max(class_results);  
        testResults = maxCount;    
    end
outpca=reshape(testResults,size(Data,1),size(Data,2)); %#ok<NASGU> % outpca为PCA特征下影像总的预测输出结果

% ==========(2)随机森林分类器==============
% nTrees = 100; %设置随机森林分类树的数量
% testResults = randomforest(nTrees, Xtr, train_label', Xts);
% outpca=reshape(testResults,size(Data,1),size(Data,2)); %#ok<NASGU> % outpca为PCA特征下影像总的预测输出结果

% ==========(3)SVM分类器==============
% verbose = true; % 输出提示信息
% % class_num; % 类别数
% classes = numN; % 类别编号
% svms = cell(class_num,1);
% kernel_type = 'polynomial'; % 设置核的类型，其他选项还包括：'RBF'和'linear'；
% [testResults] = svm(verbose,class_num,classes,svms,kernel_type,Xtr,train_label',Xts);

% outpca=reshape(testResults,size(Data,1),size(Data,2)); %#ok<NASGU> % outpca为PCA特征下影像总的预测输出结果

% ==========(4)欧式距离分类器==============
% m_hat=[];
% S_hat=[];
% Train_array = Xtr';
% % 估计最大似然参数
% for i=1:9  % 类别数为9
%     for j=1:15 %特征维度为15
%         X=Train_array(j,find(train_label==i));
%         [l,N]=size(X);
%         m=(1/N)*sum(X')';
%         S=zeros(l);
%         for k=1:N
%             S=S+(X(:,k)-m)*(X(:,k)-m)';
%         end
%         S=(1/N)*S;
%         m_hat(i, j)=m;
%         S_hat(i, j)=S;
%     end
%     m_hat(i)=m_hat(i)';S_hat(i)=S_hat(i)';
% end

% %%% 欧式距离分类器 %%%
% m_hat=m_hat';
% testResults=euclidean_distance_classifier(m_hat,Xts');
% outpca=reshape(testResults,size(Data,1),size(Data,2)); % outpca为PCA特征下影像总的预测输出结果


% ==========(5)朴素贝叶斯分类器==============
% m_hat=[];
% S_hat=[];
% Train_array = Xtr';
% % Estimate the maximum Likelihood parameters
% for i=1:9
%     for j=1:15 %特征维度为15
%         X=Train_array(j,find(train_label==i));
%         [l,N]=size(X);
%         m=(1/N)*sum(X')';
%         S=zeros(l);
%         for k=1:N
%             S=S+(X(:,k)-m)*(X(:,k)-m)';
%         end
%         S=(1/N)*S;
%         m_hat(i, j)=m;
%         S_hat(i, j)=S;
%     end
%     m_hat(i)=m_hat(i)';S_hat(i)=S_hat(i)';
% end

%%% Naive Bayes %%%
% testResults = naive_bayes_classifier(S_hat, m_hat, Xts');
% outpca=reshape(testResults,size(Data,1),size(Data,2)); % outpca为PCA特征下影像总的预测输出结果
%%
mask = trainlabels;
labels = grd;
save('labels','labels');
% 计算每一类正确率并显示，计算的是grd部分中测试数据集的分类准确率
for i=1:class_num
     accuracy(i)=size(find(outpca==labels & outpca==i & labels==i & mask==0),1)/size(find(labels==i & mask==0),1);    
end

disp('每一类的分类正确率为：')
for i=1:class_num
    disp(['第 ' num2str(i),'正确率为： ', sprintf('%.2f%',accuracy(i))]);
 
end

disp('总体分类正确率为: ')
OA=sum(outpca==labels)/sum(labels>0)  %总体分类正确率

%% 显示分类结果
[xxt,yyt]=find(grd==0);
for i=1:size(xxt,1)
   outpca(xxt(i),yyt(i))=0; %只显示测试样本
end

fig = University_colorchange(outpca);
fig = uint8(fig);
figure;
image(fig);

%% 存入分类结果
a=sprintf('Upavia%1f%',OA*100);
b='.png';
c=strcat(a,b);
imwrite(fig,c, 'png');
