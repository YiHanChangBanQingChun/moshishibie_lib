clear;
clc;

% 主程序
% 加载并分割数据
[train_data, train_labels, test_data, test_labels, unique_labels, num_classes] = load_and_split_data('D:\Users\admin\Documents\MATLAB\moshishibie_lib\上课实验代码\example\Upavia50.mat');

% 数据归一化
[train_data, test_data] = normalize_data(train_data, test_data);

% 训练贝叶斯分类器
[priors, means, covariances] = train_bayes_classifier(train_data, train_labels, unique_labels, num_classes);

% 使用贝叶斯分类器进行预测
predicted_labels = classify_bayes(test_data, priors, means, covariances, unique_labels);

% 评估贝叶斯分类器
evaluate_classification(predicted_labels, test_labels, unique_labels, '贝叶斯分类器');

% 训练SVM分类器
model = train_svm_classifier(train_data, train_labels);

% 使用SVM分类器进行预测
predicted_labels_svm = classify_svm(test_data, model);

% 评估SVM分类器
evaluate_classification(predicted_labels_svm, test_labels, unique_labels, 'SVM分类器');

% 函数定义

function [train_data, train_labels, test_data, test_labels, unique_labels, num_classes] = load_and_split_data(filepath)
    load(filepath); %#ok<LOAD>
    data = Upaviadata50;
    labels = Upavialabel;
    unique_labels = unique(labels);
    num_classes = length(unique_labels);
    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];
    for i = 1:num_classes
        class_data = data(labels == unique_labels(i), :);
        rand_indices = randperm(50);
        train_indices = rand_indices(1:40);
        test_indices = rand_indices(41:50);
        train_data = [train_data; class_data(train_indices, :)]; %#ok<AGROW>
        train_labels = [train_labels; double(unique_labels(i)) * ones(40, 1)]; %#ok<AGROW>
        test_data = [test_data; class_data(test_indices, :)]; %#ok<AGROW>
        test_labels = [test_labels; double(unique_labels(i)) * ones(10, 1)]; %#ok<AGROW>
    end
end

function [train_data_norm, test_data_norm] = normalize_data(train_data, test_data)
    mu = mean(train_data);
    sigma = std(train_data);
    sigma(sigma == 0) = 1;
    train_data_norm = (train_data - mu) ./ sigma;
    test_data_norm = (test_data - mu) ./ sigma;
end

function [priors, means, covariances] = train_bayes_classifier(train_data, train_labels, unique_labels, num_classes)
    priors = zeros(num_classes, 1);
    means = zeros(num_classes, size(train_data, 2));
    covariances = zeros(size(train_data, 2), size(train_data, 2), num_classes);
    for i = 1:num_classes
        class_data = train_data(train_labels == unique_labels(i), :);
        priors(i) = size(class_data, 1) / size(train_data, 1);
        means(i, :) = mean(class_data);
        covariances(:, :, i) = cov(class_data) + 1e-6 * eye(size(class_data, 2));
    end
end

function predicted_labels = classify_bayes(test_data, priors, means, covariances, unique_labels)
    num_classes = length(priors);
    predicted_labels = zeros(size(test_data, 1), 1);
    for i = 1:size(test_data, 1)
        posteriors = zeros(num_classes, 1);
        for j = 1:num_classes
            likelihood = mvnpdf(test_data(i, :), means(j, :), covariances(:, :, j));
            posteriors(j) = priors(j) * likelihood;
        end
        [~, max_index] = max(posteriors);
        predicted_labels(i) = unique_labels(max_index);
    end
end

function predicted_labels_svm = classify_svm(test_data, model)
    predicted_labels_svm = predict(model, test_data);
end

function model = train_svm_classifier(train_data, train_labels)
    X_train = train_data;
    y_train = train_labels;
    template = templateSVM('KernelFunction', 'linear', 'Standardize', true);
    model = fitcecoc(X_train, y_train, 'Learners', template);
end

function evaluate_classification(predicted_labels, true_labels, unique_labels, classifier_name)
    fprintf('%s分类结果:\n', classifier_name);
    accuracy = sum(predicted_labels == true_labels) / length(true_labels);
    fprintf('分类准确率: %.2f%%\n', accuracy * 100);
    confusion_matrix = confusionmat(true_labels, predicted_labels);
    disp(['混淆矩阵(', classifier_name, '):']);
    disp(array2table(confusion_matrix, 'VariableNames', strcat('预测_', string(unique_labels)), 'RowNames', strcat('真实_', string(unique_labels))));
    class_accuracies = diag(confusion_matrix) ./ sum(confusion_matrix, 2);
    for i = 1:length(unique_labels)
        fprintf('类别 %d 的正确率: %.2f%%\n', unique_labels(i), class_accuracies(i) * 100);
    end
end