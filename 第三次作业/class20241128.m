clear;
clc;

% 定义日志文件夹路径
log_folder = 'D:\Users\admin\Documents\MATLAB\moshishibie_lib\上课实验代码\example';

% 检查日志文件夹是否存在，如果不存在则创建
if ~exist(log_folder, 'dir')
    mkdir(log_folder);
end

% 获取当前时间并格式化为文件名
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
log_filename = fullfile(log_folder, ['exp_', timestamp, '.txt']);

% 打开日志文件
fid = fopen(log_filename, 'w');
if fid == -1
    error('无法创建日志文件: %s', log_filename);
end

% 输出开始记录日志
log_print(fid, '开始记录日志：%s\n', log_filename);

% 主程序

% 调用模拟函数
num_runs = 100; % 模拟次数
filepath = 'D:\Users\admin\Documents\MATLAB\moshishibie_lib\上课实验代码\example\Upavia50.mat';
k = 3; % KNN 的邻居数

simulate_classification_runs(num_runs, filepath, k, fid);

% 输出结束记录日志
log_print(fid, '结束记录日志\n');

% 关闭日志文件
fclose(fid);

% 打印日志保存路径到命令窗口
fprintf('日志文件已保存到: %s\n', log_filename);

function simulate_classification_runs(num_runs, filepath, k, fid)
    % 模拟分类运行
    % num_runs: 模拟次数
    % filepath: 数据文件路径
    % k: KNN 的邻居数
    % fid: 日志文件标识符

    % 初始化存储变量
    overall_acc = zeros(num_runs, 3); % 列1:贝叶斯, 列2:SVM, 列3:KNN
    per_class_acc = zeros(num_runs, 3, 9); % 第3维度假设有9个类别

    for run = 1:num_runs
        log_print(fid, '------------------------------------------------------------\n');
        log_print(fid, '运行 %d/%d:\n', run, num_runs);

        % 随机分割数据
        [train_data, train_labels, test_data, test_labels, unique_labels, num_classes] = load_and_split_data(filepath);

        % 数据归一化
        [train_data, test_data] = normalize_data(train_data, test_data);
        log_print(fid, '----贝叶斯分类器----\n');
        % 训练贝叶斯分类器
        [priors, means, covariances] = train_bayes_classifier(train_data, train_labels, unique_labels, num_classes);

        % 使用贝叶斯分类器进行预测
        predicted_labels = classify_bayes(test_data, priors, means, covariances, unique_labels);

        % 评估贝叶斯分类器
        [accuracy_bayes, class_acc_bayes] = evaluate_classification(predicted_labels, test_labels, unique_labels, '贝叶斯分类器', fid);
        overall_acc(run, 1) = accuracy_bayes;
        per_class_acc(run, 1, :) = class_acc_bayes';
        log_print(fid, '----SVM分类器----\n');
        % 训练SVM分类器
        model = train_svm_classifier(train_data, train_labels);

        % 使用SVM分类器进行预测
        predicted_labels_svm = classify_svm(test_data, model);

        % 评估SVM分类器
        [accuracy_svm, class_acc_svm] = evaluate_classification(predicted_labels_svm, test_labels, unique_labels, 'SVM分类器', fid);
        overall_acc(run, 2) = accuracy_svm;
        per_class_acc(run, 2, :) = class_acc_svm';
        log_print(fid, '----KNN分类器----\n');
        % 训练KNN分类器
        predicted_labels_knn = classify_knn(train_data, train_labels, test_data, k); % k=3

        % 评估KNN分类器
        [accuracy_knn, class_acc_knn] = evaluate_classification(predicted_labels_knn, test_labels, unique_labels, 'KNN分类器', fid);
        overall_acc(run, 3) = accuracy_knn;
        per_class_acc(run, 3, :) = class_acc_knn';

        log_print(fid, '\n');
    end

    plot_overall_accuracy(overall_acc, num_runs);
    plot_per_class_accuracy(per_class_acc, num_runs, 9);
end

function [train_data, train_labels, test_data, test_labels, unique_labels, num_classes] = load_and_split_data(filepath)
    % 加载并分割数据
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
        % 每个类别的数据集中有50个样本，每个类别的训练集中有40个样本，测试集中有10个样本
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
    % 归一化数据
    mu = mean(train_data);
    sigma = std(train_data);
    sigma(sigma == 0) = 1;
    train_data_norm = (train_data - mu) ./ sigma;
    test_data_norm = (test_data - mu) ./ sigma;
end

function [priors, means, covariances] = train_bayes_classifier(train_data, train_labels, unique_labels, num_classes)
    % 训练贝叶斯分类器
    priors = zeros(num_classes, 1);
    means = zeros(num_classes, size(train_data, 2));
    covariances = zeros(size(train_data, 2), size(train_data, 2), num_classes);
    % 对每个类别计算先验概率、均值和协方差矩阵
    for i = 1:num_classes
        % 获取当前类别的数据
        class_data = train_data(train_labels == unique_labels(i), :);
        % 计算先验概率、均值和协方差矩阵
        priors(i) = size(class_data, 1) / size(train_data, 1);
        means(i, :) = mean(class_data);
        % 协方差矩阵加上一个很小的对角矩阵，避免奇异矩阵。
        % 因为协方差矩阵是一个对称矩阵，所以只需要存储上三角部分即可。
        covariances(:, :, i) = cov(class_data) + 1e-6 * eye(size(class_data, 2));
    end
end

function predicted_labels = classify_bayes(test_data, priors, means, covariances, unique_labels)
    % 使用贝叶斯分类器进行预测
    num_classes = length(priors);
    predicted_labels = zeros(size(test_data, 1), 1);
    % 对每个测试样本计算后验概率，选择概率最大的类别作为预测结果
    for i = 1:size(test_data, 1)
        % 初始化后验概率为0
        posteriors = zeros(num_classes, 1);
        % 计算每个类别的后验概率
        for j = 1:num_classes
            % 计算多元高斯分布的概率密度函数
            likelihood = mvnpdf(test_data(i, :), means(j, :), covariances(:, :, j));
            posteriors(j) = priors(j) * likelihood;
        end
        % 选择概率最大的类别作为预测结果
        [~, max_index] = max(posteriors);
        % 将预测结果保存到predicted_labels中
        predicted_labels(i) = unique_labels(max_index);
    end
end

function predicted_labels_svm = classify_svm(test_data, model)
    % 使用SVM分类器进行预测
    predicted_labels_svm = predict(model, test_data);
end

function model = train_svm_classifier(train_data, train_labels)
    X_train = train_data;
    y_train = train_labels;
    % 使用SVM分类器，设置核函数为线性核函数
    % templateSVM函数是一个模板，可以设置SVM分类器的参数
    template = templateSVM('KernelFunction', 'linear', 'Standardize', true);
    % fitcecoc是多类分类器，可以处理多类别分类问题
    % 除此之外，还可以设置其他的参数，例如'OptimizeHyperparameters'、'HyperparameterOptimizationOptions'等
    model = fitcecoc(X_train, y_train, 'Learners', template);
end

function predicted_labels_knn = classify_knn(train_data, train_labels, test_data, k)
    % 使用fitcknn训练KNN模型
    mdl = fitcknn(train_data, train_labels, 'NumNeighbors', k, 'Standardize', 1);
    % 使用训练好的模型进行预测
    predicted_labels_knn = predict(mdl, test_data);
end

function [accuracy, class_accuracies] = evaluate_classification(predicted_labels, true_labels, unique_labels, classifier_name, fid)
    % 评估分类器
    log_print(fid, '%s分类结果:\n', classifier_name);

    % 计算分类准确率
    accuracy = sum(predicted_labels == true_labels) / length(true_labels);
    log_print(fid, '分类准确率: %.2f%%\n', accuracy * 100);

    % 计算混淆矩阵
    confusion_matrix = confusionmat(true_labels, predicted_labels);
    log_print(fid, '混淆矩阵(%s):\n', classifier_name);
    disp(array2table(confusion_matrix, 'VariableNames', strcat('预测_', string(unique_labels)), 'RowNames', strcat('真实_', string(unique_labels))));

    % 将混淆矩阵写入日志
    log_print(fid, '%s 混淆矩阵:\n', classifier_name);
    log_print(fid, '%s\n', mat2str(confusion_matrix));

    % 计算每个类别的正确率
    class_accuracies = diag(confusion_matrix) ./ sum(confusion_matrix, 2);
    for i = 1:length(unique_labels)
        log_print(fid, '类别 %d 的正确率: %.2f%%\n', unique_labels(i), class_accuracies(i) * 100);
    end
end

function plot_overall_accuracy(overall_acc, num_runs)
    % 绘制总体分类准确率
    figure;
    plot(1:num_runs, overall_acc(:,1)*100, '-o', 'DisplayName', '贝叶斯分类器');
    hold on;
    plot(1:num_runs, overall_acc(:,2)*100, '-s', 'DisplayName', 'SVM分类器');
    plot(1:num_runs, overall_acc(:,3)*100, '-^', 'DisplayName', 'KNN分类器');
    hold off;
    xlabel('运行次数');
    ylabel('分类准确率 (%)');
    title('三种分类方法的总体分类准确率');
    legend('Location', 'best');
    grid on;
end

function plot_per_class_accuracy(per_class_acc, num_runs, num_classes)
    % 绘制每个类别的分类准确率
    figure;
    for class_idx = 1:num_classes
        subplot(3,3,class_idx);
        plot(1:num_runs, squeeze(per_class_acc(:,1,class_idx))*100, '-o', 'DisplayName', '贝叶斯分类器');
        hold on;
        plot(1:num_runs, squeeze(per_class_acc(:,2,class_idx))*100, '-s', 'DisplayName', 'SVM分类器');
        plot(1:num_runs, squeeze(per_class_acc(:,3,class_idx))*100, '-^', 'DisplayName', 'KNN分类器');
        hold off;
        xlabel('运行次数');
        ylabel('准确率 (%)');
        title(['类别 ', num2str(class_idx), ' 的分类准确率']);
        if class_idx == 1
            legend('Location', 'best');
        end
        grid on;
    end
    sgtitle('三种分类方法的每个类别分类准确率');
end

function log_print(fid, fmt, varargin)
    % 将格式化的字符串写入日志文件
    fprintf(fid, fmt, varargin{:});
    % 同时在命令窗口中显示
    fprintf(fmt, varargin{:});
end