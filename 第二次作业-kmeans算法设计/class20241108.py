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
        plot_results(data_subset, mapped_labels, centroids, var_names, i, j, accuracy, axes[idx])

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
    ax.scatter(data_subset[:, 0], data_subset[:, 1], data_subset[:, 2], c=mapped_labels, s=50, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=200, c='red', marker='X')
    ax.set_title(f'K-means 聚类 ({var_names[0]}, {var_names[1]}, {var_names[2]})\n准确率: {accuracy:.2f}')
    ax.set_xlabel(var_names[0])
    ax.set_ylabel(var_names[1])
    ax.set_zlabel(var_names[2])
    plt.show()

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
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    mask = np.all(np.abs(data - mean) < m * std, axis=1)
    filtered_data = data[mask]
    filtered_labels = labels[mask]

    print(f'Removed {data.shape[0] - filtered_data.shape[0]} outliers')
    return filtered_data, filtered_labels, mask

def initialize_centroids(data, k, grid_density=10):
    """
    使用基于网格密度的方法初始化k-means聚类的质心。
    参数:
    data (numpy.ndarray): 数据集，每行代表一个数据点。
    k (int): 聚类的数量。
    grid_density (int): 网格密度，默认为10。
    返回:
    numpy.ndarray: 一个形状为 (k, data.shape[1]) 的数组，包含初始化的质心。
    """
    print("Initializing centroids...")
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    grid_size = (max_vals - min_vals) / grid_density
    centroids = []
    grid_counts = np.zeros([grid_density] * data.shape[1])
    for point in data:
        grid_idx = tuple(((point - min_vals) / grid_size).astype(int))
        grid_idx = tuple(min(idx, grid_density - 1) for idx in grid_idx)  # 防止越界
        grid_counts[grid_idx] += 1
    for _ in range(k):
        max_density_idx = np.unravel_index(np.argmax(grid_counts), grid_counts.shape)
        centroid = min_vals + np.array(max_density_idx) * grid_size + grid_size / 2
        centroids.append(centroid)
        grid_counts[max_density_idx] = 0
    return np.array(centroids)

def kmeans(data, k, max_iters=100, n_init=10):
    """
    对给定数据执行k-means聚类。
    参数:
    data (numpy.ndarray): 要聚类的输入数据，每行是一个数据点。
    k (int): 聚类的数量。
    max_iters (int, optional): 运行算法的最大迭代次数。默认值为100。
    n_init (int, optional): 运行算法的次数，选择最佳结果。默认值为10。

    返回:
    tuple: 包含以下内容的元组:
        - centroids (numpy.ndarray): 聚类的最终质心。
        - labels (numpy.ndarray): 每个数据点所属的聚类标签。
    """
    best_inertia = np.inf
    best_centroids = None
    best_labels = None

    for _ in range(n_init):
        centroids = initialize_centroids(data, k)
        for _ in range(max_iters):
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids

        inertia = np.sum((data - centroids[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels

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
    label_mapping = {}
    for i in range(np.max(labels) + 1):
        mask = (labels == i)
        true_label = mode(true_labels[mask], keepdims=True)[0][0]
        label_mapping[i] = true_label
    mapped_labels = np.array([label_mapping[label] for label in labels])
    accuracy = accuracy_score(true_labels, mapped_labels)
    return accuracy, mapped_labels

def plot_results(data_subset, mapped_labels, centroids, var_names, i, j, accuracy, ax):
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
    ax.scatter(data_subset[:, 0], data_subset[:, 1], c=mapped_labels, s=50, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X')
    ax.set_title(f'K-means 聚类 ({var_names[i]} vs {var_names[j]})\n准确率: {accuracy:.2f}')
    ax.set_xlabel(var_names[i])
    ax.set_ylabel(var_names[j])

def configure_plot():
    """
    - 配置全局绘图参数
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    main()