import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from mpl_toolkits.mplot3d import Axes3D

def main():
    # 读取数据，设置参数
    configure_plot()
    filename = 'D:\\Users\\admin\\Documents\\MATLAB\\moshishibie_lib\\第二次作业-kmeans算法设计\\data\\iris34.xlsx'
    k = 3
    data = pd.read_excel(filename)
    variables = data.iloc[:, 1:5].values
    true_labels = data.iloc[:, 5].values
    var_names = data.columns[1:5]

    # 去除异常值
    filtered_data, filtered_labels, mask = remove_outliers(variables, true_labels)

    # 绘制散点图
    combs = list(combinations(range(variables.shape[1]), 2))
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    all_labels = np.zeros((filtered_labels.shape[0], len(combs)), dtype=int)
    # 对每一对变量执行 K-means 聚类
    for idx, (i, j) in enumerate(combs):
        data_subset = filtered_data[:, [i, j]]

        # 执行 K-means 聚类
        centroids, labels = kmeans(data_subset, k)

        # 计算准确率
        accuracy, mapped_labels = calculate_accuracy(labels, filtered_labels)
        print(f'Accuracy for combination ({var_names[i]} vs {var_names[j]}): {accuracy:.2f}')
        all_labels[:, idx] = mapped_labels
        plot_2d_results(data_subset, mapped_labels, centroids, var_names, i, j, accuracy, axes[idx])

    # 计算总体准确率
    final_labels = mode(all_labels, axis=1, keepdims=False)[0]
    overall_accuracy = accuracy_score(filtered_labels, final_labels)
    print(f'Overall Accuracy: {overall_accuracy:.2f}')

    # 计算每个类别的准确率
    unique_labels = np.unique(filtered_labels)
    for label in unique_labels:
        label_accuracy = accuracy_score(filtered_labels[filtered_labels == label], final_labels[filtered_labels == label])
        print(f'Accuracy for label {label}: {label_accuracy:.2f}')

    # 显示图表
    for ax in axes[len(combs):]:
        fig.delaxes(ax)
    plt.tight_layout()
    plt.show()
    
    # 绘制三维图
    data_subset_3d = filtered_data[:, :3]
    centroids_3d, labels_3d = kmeans(data_subset_3d, k)
    accuracy_3d, mapped_labels_3d = calculate_accuracy(labels_3d, filtered_labels)
    plot_3d_results(data_subset_3d, mapped_labels_3d, centroids_3d, var_names[:3], accuracy_3d)

    # 保存聚类数据
    data['Cluster'] = np.nan
    data.loc[mask, 'Cluster'] = final_labels
    data.to_excel('clustered_data.xlsx', index=False)

def plot_3d_results(data_subset, mapped_labels, centroids, var_names, accuracy):
    """
    绘制 K-means 聚类结果的三维散点图。
    参数:
    data_subset (ndarray): 要绘制的数据子集，形状为 (n_samples, 3)。
    mapped_labels (ndarray): 分配给每个数据点的聚类标签数组，形状为 (n_samples,)。
    centroids (ndarray): 聚类中心的坐标，形状为 (n_clusters, 3)。
    var_names (list of str): 数据集维度对应的变量名列表。
    accuracy (float): 聚类算法的准确率。
    返回:
    None
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_subset[:, 0], data_subset[:, 1], data_subset[:, 2], c=mapped_labels, s=50, cmap='viridis', label='Data Points')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=200, c='red', marker='X', label='Centroids')
    ax.set_title(f'K-means 聚类 ({var_names[0]}, {var_names[1]}, {var_names[2]})\n准确率: {accuracy:.2f}')
    ax.set_xlabel(var_names[0])
    ax.set_ylabel(var_names[1])
    ax.set_zlabel(var_names[2])
    handles, labels = scatter.legend_elements()
    handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10, label='Centroids'))
    labels.append('Centroids')
    ax.legend(handles, labels, title="Clusters")
    plt.show()

def plot_2d_results(data_subset, mapped_labels, centroids, var_names, i, j, accuracy, ax):
    """
    - 绘制 K-means 聚类结果的二维散点图。
    - 参数:
        1. data_subset (ndarray): 要绘制的数据子集，形状为 (n_samples, 2)。
        2. mapped_labels (ndarray): 分配给每个数据点的聚类标签数组，形状为 (n_samples,)。
        3. centroids (ndarray): 聚类中心的坐标，形状为 (n_clusters, 2)。
        4. var_names (list of str): 数据集维度对应的变量名列表。
        5. i (int): 要绘制在 x 轴上的变量的索引。
        6. j (int): 要绘制在 y 轴上的变量的索引。
        7. accuracy (float): 聚类算法的准确率。
        8. ax (matplotlib.axes.Axes): 用于绘制结果的 Matplotlib Axes 对象。
    - 返回:
        None
    """
    scatter = ax.scatter(data_subset[:, 0], data_subset[:, 1], c=mapped_labels, s=50, cmap='viridis', label='Data Points')
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
    ax.set_title(f'K-means 聚类 ({var_names[i]} vs {var_names[j]})\n准确率: {accuracy:.2f}')
    ax.set_xlabel(var_names[i])
    ax.set_ylabel(var_names[j])
    handles, labels = scatter.legend_elements()
    handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10, label='Centroids'))
    labels.append('Centroids')
    ax.legend(handles, labels, title="Clusters")

def remove_outliers(data, labels, m=3):
    """
    - 根据指定的标准差数量移除数据集中的异常值。
    - 参数:
        1. data (numpy.ndarray): 输入数据数组，每行代表一个数据点，每列代表一个特征。
        2. labels (numpy.ndarray): 数据点对应的标签。
        3. m (int, 可选): 用于识别异常值的标准差数量阈值。默认值为3。
    - 返回:
    tuple: 包含以下内容的元组:
        - filtered_data (numpy.ndarray): 移除异常值后的数据数组。
        - filtered_labels (numpy.ndarray): 移除异常值后的标签数组。
        - mask (numpy.ndarray): 一个布尔数组，指示哪些数据点被保留（True）哪些被移除（False）。
    """
    print("Removing outliers...")
    
    # 计算数据集中每个特征的均值
    mean = np.mean(data, axis=0)
    
    # 计算数据集中每个特征的标准差
    std = np.std(data, axis=0)
    
    # 计算每个数据点与均值的差值，并判断是否在 m 个标准差范围内
    mask = np.all(np.abs(data - mean) < m * std, axis=1)
    
    # 根据 mask 过滤数据，保留非异常值的数据点
    filtered_data = data[mask]
    
    # 根据 mask 过滤标签，保留非异常值的数据点对应的标签
    filtered_labels = labels[mask]

    # 打印移除的异常值数量
    print(f'Removed {data.shape[0] - filtered_data.shape[0]} outliers')
    
    # 返回过滤后的数据、标签和布尔掩码
    return filtered_data, filtered_labels, mask

def initialize_centroids(data, k):
    """
    使用基于网格密度的方法初始化k-means聚类的质心。
    参数:
    data (numpy.ndarray): 数据集，每行代表一个数据点。
    k (int): 聚类的数量。
    返回:
    numpy.ndarray: 一个形状为 (k, data.shape[1]) 的数组，包含初始化的质心。
    """
    print("Initializing centroids...")
    
    # 计算数据集中每个特征的最小值
    min_vals = np.min(data, axis=0)
    
    # 计算数据集中每个特征的最大值
    max_vals = np.max(data, axis=0)
    
    # 计算网格的大小，每个特征的范围除以聚类数量k
    grid_size = (max_vals - min_vals) / k
    
    # 初始化质心列表
    centroids = []
    
    # 初始化网格计数数组，形状为 (k, k, ..., k)，维度与数据特征数相同
    grid_counts = np.zeros([k] * data.shape[1])

    # 计算每个数据点所在的网格索引
    for point in data:
        # 计算每个数据点在网格中的索引
        grid_idx = tuple(((point - min_vals) / grid_size).astype(int))
        
        # 防止索引越界，将索引限制在 [0, k-1] 范围内
        grid_idx = tuple(min(idx, k - 1) for idx in grid_idx)
        
        # 增加对应网格的计数
        grid_counts[grid_idx] += 1

    # 选择网格密度最大的k个网格中心
    for _ in range(k):
        
        # 找到最大密度的网格索引
        max_density_idx = np.unravel_index(np.argmax(grid_counts), grid_counts.shape)

        # 计算质心，质心为网格中心点
        centroid = min_vals + np.array(max_density_idx) * grid_size + grid_size / 2
        
        # 将质心添加到质心列表中
        centroids.append(centroid)
        
        # 将该网格的计数置为0，以便下次找到下一个最大密度的网格
        grid_counts[max_density_idx] = 0
    
    # 返回质心数组
    return np.array(centroids)

def kmeans(data, k, max_iters=100):
    """
    - 对给定数据执行k-means聚类。
    - 参数:
        1. data (numpy.ndarray): 要聚类的输入数据，每行是一个数据点。
        2. k (int): 聚类的数量。
        3. max_iters (int, optional): 运行算法的最大迭代次数。默认值为100。
    - 返回:
    tuple: 包含以下内容的元组:
        - centroids (numpy.ndarray): 聚类的最终质心。
        - labels (numpy.ndarray): 每个数据点所属的聚类标签。
    """
    print("Running K-means algorithm...")
    
    # 初始化质心
    centroids = initialize_centroids(data, k)
    
    # 迭代进行 K-means 聚类
    for _ in range(max_iters):
        print(f'Iteration {_ + 1}')
        
        # 计算每个数据点到每个质心的距离
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        
        # 根据距离将每个数据点分配到最近的质心
        labels = np.argmin(distances, axis=1)
        
        # 计算新的质心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # 检查质心是否收敛
        if np.all(centroids == new_centroids):
            break
        
        # 更新质心
        centroids = new_centroids
    
    # 返回最终的质心和标签
    return centroids, labels

def calculate_accuracy(labels, true_labels):
    """
    - 计算预测标签与真实标签的准确率。
    - 该函数将每个预测标签映射到同一聚类中最频繁出现的真实标签，
    - 然后计算这种映射的准确率。
    - 参数:
        1. labels (np.ndarray): 预测标签数组。
        2. true_labels (np.ndarray): 真实标签数组。
    - 返回:
    tuple: 包含以下内容的元组:
        - accuracy (float): 预测标签的准确率。
        - mapped_labels (np.ndarray): 映射到最频繁真实标签的标签数组。
    """
    # 初始化一个空字典，用于存储预测标签到真实标签的映射
    label_mapping = {}
    
    # 遍历所有预测标签的唯一值
    for i in range(np.max(labels) + 1):
        # 创建一个布尔掩码，标记当前预测标签 i 的数据点
        mask = (labels == i)
        
        # 找到当前预测标签 i 对应的真实标签中最频繁出现的标签
        true_label = mode(true_labels[mask], keepdims=True)[0][0]
        
        # 将预测标签 i 映射到最频繁出现的真实标签
        label_mapping[i] = true_label
    
    # 使用映射关系将所有预测标签转换为对应的真实标签
    mapped_labels = np.array([label_mapping[label] for label in labels])
    
    # 计算映射后的标签与真实标签之间的准确率
    accuracy = accuracy_score(true_labels, mapped_labels)
    
    # 返回准确率和映射后的标签数组
    return accuracy, mapped_labels

def configure_plot():
    """
    - 配置全局绘图参数
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    main()