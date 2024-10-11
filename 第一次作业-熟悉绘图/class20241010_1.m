clear;
clc;
close all;

% 定义高程数据
elevation = [
    1480 1500 1550 1510 1430 1300 1200 980;
    1500 1550 1600 1550 1600 1600 1600 1550;
    1500 1200 1100 1550 1600 1550 1380 1070;
    1500 1200 1100 1350 1450 1200 1150 1010;
    1390 1500 1500 1400 900 1100 1060 950;
    1320 1450 1420 1400 1300 700 900 850;
    1130 1250 1280 1230 1040 900 500 1125
    % -----------替换为1125，原数据是700__↑
];

% 定义x和y坐标，创建网格
x = 1200:400:4000;
y = 1200:400:3600;
[X, Y] = meshgrid(x, y);

% 插值，并初始化为NaN表示无数据区域
[Xq, Yq] = meshgrid(0:400:5600, 0:400:4800);
Zq = NaN(size(Xq));
Zq(Yq >= 1200 & Yq <= 3600 & Xq >= 1200 & Xq <= 4000) = interp2(X, Y, elevation, Xq(Yq >= 1200 & Yq <= 3600 & Xq >= 1200 & Xq <= 4000), Yq(Yq >= 1200 & Yq <= 3600 & Xq >= 1200 & Xq <= 4000), 'linear');

% 绘制地貌图
plot_surface(Xq, Yq, Zq, '地貌图');

% 绘制等高线图
plot_contour(Xq, Yq, Zq, '等高线图');

% 绘制地貌图
function plot_surface(Xq, Yq, Zq, plot_title)
    figure;
    surf(Xq, Yq, Zq, 'EdgeColor', 'none');
    title(plot_title);
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('高程 (m)');
    colorbar;
    colormap('spring');
    hold on;
    surf(Xq, Yq, zeros(size(Zq)), 'FaceColor', 'none', 'EdgeColor', [0 0 0], 'LineStyle', '--');
    hold off;
end

% 绘制等高线图
function plot_contour(Xq, Yq, Zq, plot_title)
    figure;
    contour(Xq, Yq, Zq, 'ShowText', 'on');
    title(plot_title);
    xlabel('X (m)');
    ylabel('Y (m)');
    colorbar;
    colormap('spring');
    hold on;
    surf(Xq, Yq, zeros(size(Zq)), 'FaceColor', 'none', 'EdgeColor', [0 0 0], 'LineStyle', '--');
    hold off;
end