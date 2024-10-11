import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def configure_plot():
    """
    配置全局绘图参数
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def create_grid(x_range, y_range, step):
    """
    创建网格

    参数:
    - x_range (tuple): x轴范围
    - y_range (tuple): y轴范围
    - step (int): 步长

    返回:
    - tuple: 返回网格
    """
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    return np.meshgrid(x, y)

def interpolate_data(X, Y, elevation, Xq, Yq):
    """
    插值数据

    参数:
    - X (ndarray): 原始网格的x坐标
    - Y (ndarray): 原始网格的y坐标
    - elevation (ndarray): 高程数据
    - Xq (ndarray): 插值网格的x坐标
    - Yq (ndarray): 插值网格的y坐标

    返回:
    - ndarray: 插值后的高程数据
    """
    Zq = np.full(Xq.shape, np.nan)
    mask = (Yq >= 1200) & (Yq <= 3600) & (Xq >= 1200) & (Xq <= 4000)
    Zq[mask] = griddata((X.flatten(), Y.flatten()), elevation.flatten(), (Xq[mask], Yq[mask]), method='linear')
    return Zq

def plot_surface(Xq, Yq, Zq, plot_title):
    """
    绘制地貌图

    参数:
    - Xq (ndarray): 插值网格的x坐标
    - Yq (ndarray): 插值网格的y坐标
    - Zq (ndarray): 插值后的高程数据
    - plot_title (str): 图表标题
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xq, Yq, Zq, cmap='spring', edgecolor='none')
    ax.set_title(plot_title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('高程 (m)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.plot_wireframe(Xq, Yq, np.zeros_like(Zq), color='gray', linestyle='--')
    contour = ax.contour(Xq, Yq, Zq, zdir='z', offset=np.min(Zq), cmap='spring', linestyles='--')
    ax.clabel(contour, inline=True, fontsize=8, colors='black', inline_spacing=5)
    plt.show()

def plot_contour(Xq, Yq, Zq, plot_title):
    """
    绘制等高线图

    参数:
    - Xq (ndarray): 插值网格的x坐标
    - Yq (ndarray): 插值网格的y坐标
    - Zq (ndarray): 插值后的高程数据
    - plot_title (str): 图表标题
    """
    fig, ax = plt.subplots()
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(0, 5601, 400))
    ax.set_yticks(np.arange(0, 4801, 400))
    ax.set_xticklabels(np.arange(0, 5601, 400))
    ax.set_yticklabels(np.arange(0, 4801, 400))
    ax.tick_params(axis='x', which='both', rotation=45)
    ax.tick_params(axis='y', which='both', rotation=45)
    contour = ax.contour(Xq, Yq, Zq, levels=np.arange(np.nanmin(Zq), np.nanmax(Zq), 100), cmap='spring')
    ax.set_title(plot_title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    fig.colorbar(contour)
    ax.clabel(contour, inline=False, fontsize=12, colors='black', inline_spacing=1)
    plt.show()

def main():
    """
    主函数，执行绘图流程
    """
    configure_plot()

    # 定义高程数据
    elevation = np.array([
        [1480, 1500, 1550, 1510, 1430, 1300, 1200, 980],
        [1500, 1550, 1600, 1550, 1600, 1600, 1600, 1550],
        [1500, 1200, 1100, 1550, 1600, 1550, 1380, 1070],
        [1500, 1200, 1100, 1350, 1450, 1200, 1150, 1010],
        [1390, 1500, 1500, 1400, 900, 1100, 1060, 950],
        [1320, 1450, 1420, 1400, 1300, 700, 900, 850],
        [1130, 1250, 1280, 1230, 1040, 900, 500, 1125]
    ])

    X, Y = create_grid((1200, 4001), (1200, 3601), 400)
    Xq, Yq = create_grid((0, 5601), (0, 4801), 400)
    Zq = interpolate_data(X, Y, elevation, Xq, Yq)
    plot_surface(Xq, Yq, Zq, '地貌图')
    plot_contour(Xq, Yq, Zq, '等高线图')

if __name__ == '__main__':
    main()
