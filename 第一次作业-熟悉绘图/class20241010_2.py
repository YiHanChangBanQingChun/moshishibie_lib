import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def configure_plot():
    """
    配置全局绘图参数
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def plot_heart(ax, a, b, c, d, e):
    """
    绘制心形线
    - ax (Axes): 2D 子图的轴对象
    - a, b, c, d, e (float): 心形线的系数
    """
    t = np.linspace(0, 2 * np.pi, 1000)
    x = a * np.sin(t)**3
    y = b * np.cos(t) - c * np.cos(2 * t) - d * np.cos(3 * t) - e * np.cos(4 * t)
    ax.plot(x, y, 'r', linewidth=2)
    ax.grid(True)  # 添加格栅
    ax.set_title('心形线')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.legend(['心形线'])
    ax.axis('equal')  # 定制坐标，使得x轴和y轴的比例相同

def plot_saddle(ax, xmin=-2, xmax=2, ymin=-2, ymax=2, x_num=50, y_num=50):
    """
    绘制马鞍面
    - ax (Axes3D): 3D 子图的轴对象
    - xmin (float): x轴最小值, 默认为-2
    - xmax (float): x轴最大值, 默认为2
    - ymin (float): y轴最小值, 默认为-2
    - ymax (float): y轴最大值, 默认为2
    - x_num (int): x轴网格数量, 默认为50
    - y_num (int): y轴网格数量, 默认为50
    """
    x = np.linspace(xmin, xmax, x_num)
    y = np.linspace(ymin, ymax, y_num)
    x, y = np.meshgrid(x, y)
    z = x**2 - y**2
    surf = ax.plot_surface(x, y, z, cmap='spring')
    ax.grid(True)  # 添加格栅
    ax.set_title('马鞍面')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    plt.colorbar(surf, ax=ax)
    ax.legend(['马鞍面'])

def generate_formula_text(coeff_heart, coeff_saddle):
    """
    生成公式文本
    - coeff_heart (list): 心形线的系数
    - coeff_saddle (list): 马鞍面的参数
    """
    a = coeff_heart
    b = coeff_saddle
    heart_formula = f"x={a[0]}*sin(t)^3,y={a[1]}*cos(t)-{a[2]}*cos(2t)-{a[3]}*cos(3t)-{a[4]}*cos(4t)"
    saddle_formula = f"z=x^2-y^2\n马鞍面范围是x:[{b[0]},{b[1]}],y:[{b[2]},{b[3]}]"
    return f"心形线公式:{heart_formula}\n马鞍面公式:{saddle_formula}"

def main():
    """主函数"""
    configure_plot()  # 配置全局绘图参数

    # 创建一个新的图形窗口，并设置其大小
    fig = plt.figure(figsize=(16, 8))

    # 在第一个子图中绘制心形线
    ax1 = fig.add_subplot(1, 2, 1)
    coefficients_heart = [64, 48, 20, 18, 5]  # 自定义心形线的系数(常规心形线参数为[16, 13, 5, 2, 1])
    plot_heart(ax1, *coefficients_heart)

    # 在第二个子图中绘制马鞍面
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    coefficients_saddle = [-5, 3, -4, 7, 60, 180]  # 自定义马鞍面的参数(常规马鞍面参数为[-2, 2, -2, 2])
    plot_saddle(ax2, *coefficients_saddle)

    # 添加公式文本
    formula_text = generate_formula_text(coefficients_heart, coefficients_saddle)
    fig.text(0.5, 0.01, formula_text, ha='center', fontsize=12)

    plt.show()

if __name__ == '__main__':
    main()