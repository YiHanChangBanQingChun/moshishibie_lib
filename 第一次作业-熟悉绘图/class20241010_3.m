clear;
clc;
close all;

% 探究圆柱与球体接触状况
% 注意： D的范围需要小于0
plot_sphere_and_cylinder(0, 0, 0, -2785); % 题目要求的参数信息

function plot_sphere_and_cylinder(A, B, C, D)
    % 简要判断球体是否存在
    if D >= 0
        error('D 必须小于 0');
    end
        
    % 定义球面和圆柱的参数
    r_sphere = sqrt((A^2 + B^2 + C^2 - 4*D) / 4); % 计算球的半径
    r_cylinder = 0.5 * r_sphere; % 圆柱的半径，根据 x^2 + y^2 = r x 重写得出
    h_cylinder = 2 * r_sphere; % 圆柱的高度，取为球的直径

    % 计算圆柱中心位置
    x_center = r_cylinder;

    % 生成球面的网格数据
    [theta, phi] = meshgrid(linspace(0, 2*pi, 50), linspace(0, pi, 50));
    x_sphere = r_sphere * sin(phi) .* cos(theta);
    y_sphere = r_sphere * sin(phi) .* sin(theta);
    z_sphere = r_sphere * cos(phi);

    % 生成圆柱的网格数据
    [theta_cylinder, z_cylinder] = meshgrid(linspace(0, 2*pi, 50), linspace(-h_cylinder/2, h_cylinder/2, 50));
    x_cylinder = x_center + r_cylinder * cos(theta_cylinder); % 平移圆柱中心到 (x_center, 0)
    y_cylinder = r_cylinder * sin(theta_cylinder);

    % 创建一个新的图形窗口，绘制球面和圆柱
    figure;
    surf(x_sphere, y_sphere, z_sphere, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'cyan');
    hold on;
    surf(x_cylinder, y_cylinder, z_cylinder, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'magenta');
    colormap(spring);
    
    % 设置图形属性
    axis equal;
    xlabel('X轴');
    ylabel('Y轴');
    zlabel('Z轴');
    title('球面和圆柱的交集区域');
    grid on;
    camlight('right');
    lighting phong;
    view(30, 30);
    
    % 添加图例
    legend({'球面', '圆柱'}, 'Location', 'best');
end