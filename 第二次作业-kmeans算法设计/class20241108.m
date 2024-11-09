clc;
clear all;

% 读取 .xlsx 文件
filename = 'D:\Users\admin\Documents\MATLAB\moshishibie_lib\第二次作业-kmeans算法设计\data\iris34.xlsx'; % 替换为你的 .xlsx 文件的绝对路径
data = readtable(filename, 'VariableNamingRule', 'preserve');

% 提取需要绘制的列
variables = data{:, 2:5}; % 提取第2, 3, 4, 5列

% 获取变量名
varNames = data.Properties.VariableNames(2:5);

% 绘制散点图矩阵
figure;
gplotmatrix(variables, [], [], 'b', 'o', [], 'on', '', varNames);

% 设置中文显示
set(gca, 'FontName', 'SimHei');