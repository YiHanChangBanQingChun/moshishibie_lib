import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def configure_plot():
    """
    配置全局绘图参数
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def plot_sphere_and_cylinder(A, B, C, D):
    """
    绘制球面和圆柱的交集区域
    - A, B, C, D (float): 球面方程的系数
    - D 必须小于 0
    """
    # 简要判断球体是否存在
    if D >= 0:
        raise ValueError('D 必须小于 0')
    
    # 定义球面和圆柱的参数
    r_sphere = np.sqrt((A**2 + B**2 + C**2 - 4*D) / 4)  # 计算球的半径
    r_cylinder = 0.5 * r_sphere  # 圆柱的半径，根据 x^2 + y^2 = r x 重写得出
    h_cylinder = 2 * r_sphere  # 圆柱的高度，取为球的直径

    # 计算圆柱中心位置
    x_center = r_cylinder

    # 生成球面的网格数据
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 50), 
                             np.linspace(0, np.pi, 50))
    x_sphere = r_sphere * np.sin(phi) * np.cos(theta)
    y_sphere = r_sphere * np.sin(phi) * np.sin(theta)
    z_sphere = r_sphere * np.cos(phi)

    # 生成圆柱的网格数据
    theta_cylinder, z_cylinder = np.meshgrid(np.linspace(0, 2*np.pi, 50), 
                                             np.linspace(-h_cylinder/2, h_cylinder/2, 50))
    x_cylinder = x_center + r_cylinder * np.cos(theta_cylinder)  # 平移圆柱中心到 (x_center, 0)
    y_cylinder = r_cylinder * np.sin(theta_cylinder)

    # 创建一个新的图形窗口
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制球面
    ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                    color='cyan', alpha=0.5, edgecolor='none')

    # 绘制圆柱
    ax.plot_surface(x_cylinder, y_cylinder, z_cylinder, 
                    color='magenta', alpha=0.5, edgecolor='none')

    # 设置图形属性
    ax.set_box_aspect([1, 1, 1])  # 设置坐标轴比例
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('球面和圆柱的交集区域')
    ax.grid(True)

    # 添加光照和视角以增强视觉效果
    ax.view_init(elev=30, azim=30)

    # 添加图例
    ax.legend(['球面', '圆柱'], loc='best')

    plt.show()

def main():
    configure_plot()
    plot_sphere_and_cylinder(0, 0, 0, -20031125)

if __name__ == '__main__':
    main()