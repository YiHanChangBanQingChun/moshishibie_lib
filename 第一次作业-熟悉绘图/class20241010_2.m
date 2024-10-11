clear;
clc;
close all;

% 创建一个新的图形窗口，并设置其大小
figure('Position', [100, 100, 1620, 550]);

% 在第一个子图中绘制心形线
subplot(1, 2, 1);
coefficients = [60, 40, 20, 10, 5]; % 自定义心形线的系数(常规心形线参数为[16, 13, 5, 2, 1])
plot_heart(coefficients);

% 在第二个子图中绘制马鞍面
subplot(1, 2, 2);
plot_saddle();

% 绘制心形线的函数
function plot_heart(coeff)
    t = linspace(0, 2*pi, 1000);
    x = coeff(1) * sin(t).^3;
    y = coeff(2) * cos(t) - coeff(3) * cos(2*t) - coeff(4) * cos(3*t) - coeff(5) * cos(4*t);
    plot(x, y, 'r', 'LineWidth', 2);
    grid on; % 添加格栅
    title('心形线');
    xlabel('X轴');
    ylabel('Y轴');
    legend('心形线');
    axis equal; % 定制坐标，使得x轴和y轴的比例相同
end

% 绘制马鞍面的函数
function plot_saddle()
    [x, y] = meshgrid(linspace(-2, 2, 50));
    z = x.^2 - y.^2;
    surf(x, y, z);
    grid on; % 添加格栅
    title('马鞍面');
    xlabel('X轴');
    ylabel('Y轴');
    zlabel('Z轴');
    colorbar;
    colormap('spring');
    legend('马鞍面');
end