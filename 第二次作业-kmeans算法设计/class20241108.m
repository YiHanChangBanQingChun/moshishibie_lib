clc;

main();

function main()
    % 读取数据，设置参数
    configure_plot();
    filename = 'D:\Users\admin\Documents\MATLAB\moshishibie_lib\第二次作业-kmeans算法设计\data\iris34.xlsx';
    k = 3;
    data = readtable(filename, 'VariableNamingRule', 'preserve');
    variables = table2array(data(:, 2:5));
    true_labels = table2array(data(:, 6));
    var_names = data.Properties.VariableNames(2:5);

    % 去除异常值
    [filtered_data, filtered_labels, mask] = remove_outliers(variables, true_labels);

    % 绘制散点图
    combs = nchoosek(1:size(variables, 2), 2);
    figure;
    all_labels = zeros(size(filtered_labels, 1), size(combs, 1));
    % 对每一对变量执行 K-means 聚类
    for idx = 1:size(combs, 1)
        i = combs(idx, 1);
        j = combs(idx, 2);
        data_subset = filtered_data(:, [i, j]);

        % 执行 K-means 聚类
        [centroids, labels] = kmeans(data_subset, k);

        % 计算准确率
        [accuracy, mapped_labels] = calculate_accuracy(labels, filtered_labels);
        fprintf('Accuracy for combination (%s vs %s): %.2f\n', var_names{i}, var_names{j}, accuracy);
        all_labels(:, idx) = mapped_labels;
        subplot(2, 3, idx);
        plot_2d_results(data_subset, mapped_labels, centroids, var_names, i, j, accuracy);
    end

    % 计算总体准确率
    final_labels = mode(all_labels, 2);
    overall_accuracy = sum(filtered_labels == final_labels) / length(filtered_labels);
    fprintf('Overall Accuracy: %.2f\n', overall_accuracy);

    % 计算每个类别的准确率
    unique_labels = unique(filtered_labels);
    for label = unique_labels'
        label_accuracy = sum(filtered_labels(filtered_labels == label) == final_labels(filtered_labels == label)) / sum(filtered_labels == label);
        fprintf('Accuracy for label %d: %.2f\n', label, label_accuracy);
    end

    % 绘制三维图
    data_subset_3d = filtered_data(:, 1:3);
    [centroids_3d, labels_3d] = kmeans(data_subset_3d, k);
    [accuracy_3d, mapped_labels_3d] = calculate_accuracy(labels_3d, filtered_labels);
    figure;
    plot_3d_results(data_subset_3d, mapped_labels_3d, centroids_3d, var_names(1:3), accuracy_3d);

    % 保存聚类数据
    data.Cluster = nan(height(data), 1);
    data.Cluster(mask) = final_labels;
    writetable(data, 'D:\Users\admin\Documents\MATLAB\moshishibie_lib\第二次作业-kmeans算法设计\data\clustered_iris34_m.xlsx');
end

function configure_plot()
    set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
    set(groot, 'defaultAxesTickLabelInterpreter', 'none');
end

function [filtered_data, filtered_labels, mask] = remove_outliers(data, labels, m)
    if nargin < 3
        m = 3;
    end
    fprintf('Removing outliers...\n');
    mean_vals = mean(data);
    std_vals = std(data);
    mask = all(abs(data - mean_vals) < m * std_vals, 2);
    filtered_data = data(mask, :);
    filtered_labels = labels(mask);
    fprintf('Removed %d outliers\n', size(data, 1) - size(filtered_data, 1));
end

function centroids = initialize_centroids(data, k)
    fprintf('Initializing centroids...\n');
    min_vals = min(data);
    max_vals = max(data);
    grid_size = (max_vals - min_vals) / k;
    centroids = [];
    grid_counts = zeros(repmat(k, 1, size(data, 2)));

    for i = 1:size(data, 1)
        grid_idx = min(floor((data(i, :) - min_vals) ./ grid_size) + 1, k);
        grid_counts(sub2ind(size(grid_counts), grid_idx)) = grid_counts(sub2ind(size(grid_counts), grid_idx)) + 1;
    end

    for i = 1:k
        [~, max_idx] = max(grid_counts(:));
        grid_idx = cell(1, size(data, 2));
        [grid_idx{1:size(data, 2)}] = ind2sub(size(grid_counts), max_idx);
        grid_idx = cell2mat(grid_idx);
        centroid = min_vals + (grid_idx - 0.5) .* grid_size;
        centroids = [centroids; centroid];
        grid_counts(max_idx) = 0;
    end
end

function [centroids, labels] = kmeans(data, k, max_iters)
    if nargin < 3
        max_iters = 100;
    end
    fprintf('Running K-means algorithm...\n');
    centroids = initialize_centroids(data, k);

    for iter = 1:max_iters
        fprintf('Iteration %d\n', iter);
        distances = pdist2(data, centroids);
        [~, labels] = min(distances, [], 2);
        new_centroids = arrayfun(@(i) mean(data(labels == i, :), 1), 1:k, 'UniformOutput', false);
        new_centroids = cell2mat(new_centroids');
        if isequal(centroids, new_centroids)
            break;
        end
        centroids = new_centroids;
    end
end

function [accuracy, mapped_labels] = calculate_accuracy(labels, true_labels)
    label_mapping = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
    for i = 0:max(labels)
        mask = (labels == i);
        true_label = mode(true_labels(mask));
        label_mapping(i) = true_label;
    end
    mapped_labels = arrayfun(@(x) label_mapping(x), labels);
    accuracy = sum(mapped_labels == true_labels) / length(true_labels);
end

function plot_2d_results(data_subset, mapped_labels, centroids, var_names, i, j, accuracy)
    % 绘制 K-means 聚类结果的二维散点图
    gscatter(data_subset(:, 1), data_subset(:, 2), mapped_labels, 'rgb', 'o', 8);
    hold on;
    scatter(centroids(:, 1), centroids(:, 2), 200, 'k', 'x', 'LineWidth', 2);
    title(sprintf('K-means 聚类 (%s vs %s)\n准确率: %.2f', var_names{i}, var_names{j}, accuracy), 'Interpreter', 'none');
    xlabel(var_names{i}, 'Interpreter', 'none');
    ylabel(var_names{j}, 'Interpreter', 'none');
    legend({'类别 1', '类别 2', '类别 3', '中心'}, 'Location', 'best');
    hold off;
end

function plot_3d_results(data_subset, mapped_labels, centroids, var_names, accuracy)
    % 绘制 K-means 聚类结果的三维散点图
    scatter3(data_subset(:, 1), data_subset(:, 2), data_subset(:, 3), 36, mapped_labels, 'filled');
    hold on;
    scatter3(centroids(:, 1), centroids(:, 2), centroids(:, 3), 200, 'k', 'x', 'LineWidth', 2);
    title(sprintf('K-means 聚类 (%s, %s, %s)\n准确率: %.2f', var_names{1}, var_names{2}, var_names{3}, accuracy), 'Interpreter', 'none');
    xlabel(var_names{1}, 'Interpreter', 'none');
    ylabel(var_names{2}, 'Interpreter', 'none');
    zlabel(var_names{3}, 'Interpreter', 'none');
    legend({'数据点', '中心'}, 'Location', 'best');
    hold off;
end