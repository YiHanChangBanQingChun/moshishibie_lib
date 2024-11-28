clear;
clc;

% 加载数据
load('D:\Users\admin\Documents\MATLAB\moshishibie_lib\上课实验代码\example\Upavia50.mat');

% 提取数据和标签
data = Upaviadata50;
labels = Upavialabel;

% 随机选择 360 行作为训练样本，90 行作为测试样本
num_samples = size(data, 1);
rand_indices = randperm(num_samples);
train_indices = rand_indices(1:360);
test_indices = rand_indices(361:450);

train_data = data(train_indices, :);
train_labels = labels(train_indices);
test_data = data(test_indices, :);
test_labels = labels(test_indices);

% 计算每个类别的先验概率和均值、协方差矩阵
unique_labels = unique(train_labels);
num_classes = length(unique_labels);
priors = zeros(num_classes, 1);
means = zeros(num_classes, size(train_data, 2));
covariances = zeros(size(train_data, 2), size(train_data, 2), num_classes);

for i = 1:num_classes
    class_data = train_data(train_labels == unique_labels(i), :);
    priors(i) = size(class_data, 1) / size(train_data, 1);
    means(i, :) = mean(class_data);
    covariances(:, :, i) = cov(class_data) + 1e-6 * eye(size(class_data, 2)); % 添加小的正数到对角线
end

% 对测试样本进行分类
predicted_labels = zeros(size(test_labels));
for i = 1:length(test_labels)
    predicted_labels(i) = bayes_classifier(test_data(i, :), priors, means, covariances, unique_labels);
end

% 打印每个测试样本的预测标签和真实标签
fprintf('测试样本的预测标签和真实标签:\n');
for i = 1:length(test_labels)
    fprintf('样本 %d: 预测标签 = %d, 真实标签 = %d\n', i, predicted_labels(i), test_labels(i));
end

% 计算分类准确率
accuracy = sum(predicted_labels == test_labels) / length(test_labels);
fprintf('分类准确率: %.2f%%\n', accuracy * 100);

% 贝叶斯分类器
function label = bayes_classifier(sample, priors, means, covariances, unique_labels)
    num_classes = length(priors);
    posteriors = zeros(num_classes, 1);
    
    for i = 1:num_classes
        likelihood = mvnpdf(sample, means(i, :), covariances(:, :, i));
        posteriors(i) = priors(i) * likelihood;
    end
    
    [~, max_index] = max(posteriors);
    label = unique_labels(max_index);
end