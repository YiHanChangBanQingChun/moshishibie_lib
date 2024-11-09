clc;
clear all;

% 加载训练数据集
load('data.mat'); % 替换为你的训练数据集文件名

% 查看加载的变量
whos;

% 假设数据存储在变量名为 'data' 中
% 提取自变量和因变量
X_train = data(:, 2:4); % 自变量
y_train = data(:, 5); % 因变量

% 分割训练数据集
X_train = X_train(11:end, :);
y_train = y_train(11:end);

% 添加截距项
X_train = [ones(size(X_train, 1), 1), X_train];

% 计算回归系数 (正规方程)
beta = (X_train' * X_train) \ (X_train' * y_train);

% 加载测试数据集
load('testsample.mat'); % 替换为你的测试数据集文件名

% 查看加载的变量
whos;

% 假设数据存储在变量名为 'testsample' 中
% 提取自变量和因变量
X_test = testsample(:, 2:4); % 自变量
y_test = testsample(:, 5); % 因变量

% 用训练集的前 11 行替换测试集的前 11 行
X_test(1:11, :) = data(1:11, 2:4);
y_test(1:11) = data(1:11, 5);

% 添加截距项
X_test = [ones(size(X_test, 1), 1), X_test];

% 线性回归预测
y_pred_linear = X_test * beta;

% 计算均方误差 (MSE)
mse_linear = mean((y_test - y_pred_linear).^2);

% 显示结果
disp('回归系数:');
disp(beta);
disp('线性回归均方误差:');
disp(mse_linear);

% 生成预测值和误差
predictions = [y_test, y_pred_linear, y_test - y_pred_linear];

% 显示预测值和误差
disp('预测值和误差:');
disp(predictions);

% 将预测值和误差添加到测试数据集后
testsample_with_predictions = [testsample, y_pred_linear, y_test - y_pred_linear];

% 保存为新文件
save('testsample_with_predictions.mat', 'testsample_with_predictions');

% 绘制线性回归拟合图
figure;
plot(1:length(y_test), y_test, 'bo-', 'DisplayName', '实际值');
hold on;
plot(1:length(y_test), y_pred_linear, 'r*-', 'DisplayName', '线性回归预测值');
title('线性回归拟合图');
xlabel('样本编号');
ylabel('目标值');
legend('show');
grid on;
set(gca, 'FontName', 'SimHei');

% 显示数据点的数值
for i = 1:length(y_test)
    text(i, y_test(i), num2str(y_test(i)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    text(i, y_pred_linear(i), num2str(y_pred_linear(i)), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');
end

% 在图上方显示回归方程
equation_str = sprintf('y = %.4f + %.4f*x1 + %.4f*x2 + %.4f*x3', beta(1), beta(2), beta(3), beta(4));
annotation('textbox', [0.15, 0.01, 0.1, 0.05], 'String', equation_str, 'FitBoxToText', 'on', 'HorizontalAlignment', 'center', 'FontName', 'SimHei');

% 绘制每个变量与 y 的关系的散点图
figure;
for i = 1:3
    subplot(3, 1, i);
    scatter(X_test(:, i+1), y_test, 'bo'); % 注意这里的 i+1，因为 X_test 包含了截距项
    title(['变量 ', num2str(i), ' 与 y 的关系']);
    xlabel(['变量 ', num2str(i)]);
    ylabel('y');
    grid on;
    set(gca, 'FontName', 'SimHei');
end