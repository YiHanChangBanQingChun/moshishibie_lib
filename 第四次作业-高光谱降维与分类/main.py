import os
import time
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
plt.ion()  # 开启交互模式
import datetime
import sys
from functools import wraps

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

class DualLogger:
    def __init__(self, *log_files):
        self.log_files = log_files
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        for f in self.log_files:
            f.write(message)

    def flush(self):
        self.terminal.flush()
        for f in self.log_files:
            f.flush()

def log_prints(func):
    """
    装饰器，用于记录函数运行时的所有print输出到日志文件，同时在终端显示。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取运行文件夹路径
        run_output_path = kwargs.get('run_output_path') or os.getenv('RUN_OUTPUT_PATH', '.')
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        method_abbreviation = ''.join([word[0].upper() for word in func.__name__.split('_')])[:3]
        log_file = os.path.join(run_output_path, f"LOG_{method_abbreviation}_{timestamp}.txt")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = DualLogger(original_stdout, f)
            try:
                result = func(*args, **kwargs)
            finally:
                sys.stdout = original_stdout
        print(f"运行日志已保存至 {log_file}")
        return result
    return wrapper

def timer(func):
    """
    计时器装饰器，用于测量函数的执行时间。

    参数：
    func (callable): 被装饰的函数。

    返回：
    callable: 包装后的函数。
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result
    return wrapper

def get_mat_variable_info(mat_file_path, variable_name=None):
    """
    读取 .mat 文件并获取指定变量的详细信息。如果不传入变量名称，则默认读取所有变量的信息。

    参数：
    mat_file_path (str): .mat 文件的路径。
    variable_name (str, optional): 需要获取信息的变量名称。如果为 None，则读取所有变量的信息。

    返回：
    dict: 包含变量的详细信息，包括形状、数据类型和数据。
    """
    # 读取 .mat 文件
    mat_data = scipy.io.loadmat(mat_file_path)

    # 如果没有传入变量名称，默认读取所有变量的信息
    if variable_name is None:
        variable_info = {}
        for var_name in mat_data:
            if not var_name.startswith('__'):
                variable_info[var_name] = {
                    'shape': mat_data[var_name].shape,
                    'dtype': mat_data[var_name].dtype,
                    'data': mat_data[var_name]
                }
        return variable_info

    # 检查变量是否存在
    if variable_name not in mat_data:
        raise ValueError(f"变量 '{variable_name}' 不存在于文件中。")

    # 获取变量的数据
    variable_data = mat_data[variable_name]

    # 返回变量的详细信息
    variable_info = {
        'shape': variable_data.shape,
        'dtype': variable_data.dtype,
        'data': variable_data
    }
    return variable_info

def getlabeled(groundtruth, k):
    """
    随机选择高光谱数据的标记训练样本。

    参数：
    groundtruth (np.ndarray): 原始高光谱数据的地面实况标签矩阵。
    k (int): 每个类别中选择的样本数量。

    返回：
    tuple: (trainlabels, testlabels)
    trainlabels (np.ndarray): 训练样本的标签矩阵。
    testlabels (np.ndarray): 测试样本的标签矩阵。
    """
    labeled = groundtruth.copy()
    unique_classes = np.unique(groundtruth)
    unique_classes = unique_classes[unique_classes > 0]  # 排除背景类别

    for cls in unique_classes:
        # 获取当前类别的所有索引
        indices = np.argwhere(groundtruth == cls)
        # 打乱索引顺序
        np.random.shuffle(indices)
        # 选择训练样本和测试样本
        if len(indices) > k:
            train_idx = indices[:k]
            test_idx = indices[k:]
        else:
            train_idx = indices
            test_idx = np.array([]).reshape(0, 2)
        # 将测试样本位置的标签置为0
        for idx in test_idx:
            labeled[idx[0], idx[1]] = 0

    trainlabels = labeled
    testlabels = groundtruth - trainlabels

    return trainlabels, testlabels

def load_and_preprocess_data(mat_file_path):
    """
    加载数据并进行预处理。

    参数：
    mat_file_path (str): .mat 文件的路径。

    返回：
    tuple: (pavia, rgb_pavia, test_pavia, X)
        - pavia (np.ndarray): 原始高光谱数据。
        - rgb_pavia (np.ndarray): RGB 影像数据。
        - test_pavia (np.ndarray): 测试集标签。
        - X (np.ndarray): 原始数据重塑为二维数组。
    """
    # 加载数据
    data_info = get_mat_variable_info(mat_file_path)
    pavia = data_info['Pavia']['data'].astype(np.float64)
    rgb_pavia = data_info['RGB_Pavia']['data']
    test_pavia = data_info['Test_Pavia']['data']
    # 将数据重塑为二维数组（像素数 x 波段数）
    X = pavia.reshape(-1, pavia.shape[2])
    return pavia, rgb_pavia, test_pavia, X

def calculate_class_statistics(test_pavia):
    """
    统计类别数量和每类样本数量。

    参数：
    test_pavia (np.ndarray): 测试集标签。

    返回：
    tuple: (class_num, T)
        - class_num (int): 类别数量。
        - T (list): 每个类别的样本数量列表。
    """
    numlabeled = test_pavia[test_pavia > 0]
    numN = np.unique(numlabeled)
    num = len(numN)
    T = []
    for i in numN:
        N_i = np.sum(test_pavia == i)
        T.append(N_i)
    class_num = num
    return class_num, T

def plot_rgb_image(rgb_pavia, run_output_path):
    """
    显示并保存 RGB 影像。
    
    参数：
    rgb_pavia (np.ndarray): RGB 影像数据。
    run_output_path (str): 运行文件夹路径，用于保存图片。
    """
    plt.figure()
    plt.imshow(rgb_pavia)
    plt.title('RGB_Pavia 影像')
    plt.axis('off')
    plt.show()
    
    # 保存图片到运行文件夹
    filename = os.path.join(run_output_path, 'RGB_Pavia_影像.png')
    plt.imsave(filename, rgb_pavia)
    print(f"RGB 影像已保存至 {filename}")

def plot_cumulative_explained_variance(eig_values, run_output_path):
    """
    绘制并保存累计解释方差曲线。
    
    参数：
    eig_values (np.ndarray): 特征值数组。
    run_output_path (str): 运行文件夹路径，用于保存图片。
    """
    explained_variance_ratio = eig_values / np.sum(eig_values)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    plt.figure()
    plt.plot(np.arange(1, len(cumulative_explained_variance) + 1),
             cumulative_explained_variance * 100, marker='o')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差百分比 (%)')
    plt.title('PCA 累计解释方差曲线')
    plt.grid(True)
    plt.show()
    
    # 保存图片到运行文件夹
    filename = os.path.join(run_output_path, 'PCA_累计解释方差曲线.png')
    plt.savefig(filename)
    plt.close()
    print(f"PCA 累计解释方差曲线已保存至 {filename}")
    
    return cumulative_explained_variance

def plot_validation_results(labels, title="验证集正确结果", filepath='Validation_Truth.png'):
    """
    绘制验证集的正确结果图像，并保存为文件。
    
    参数：
    labels (np.ndarray): 真实标签矩阵，形状为 (行数, 列数)。
    title (str, optional): 图像标题。默认值为 "验证集正确结果"。
    filepath (str, optional): 保存图像的完整文件路径。默认值为 "Validation_Truth.png"。
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(labels, cmap='jet')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    # 保存图片到指定路径
    plt.imsave(filepath, labels, cmap='jet')
    print(f"验证集正确结果已保存至 {filepath}")

def PCA(data, n_components):
    """
    主成分分析（PCA）

    参数：
    data (numpy.ndarray): 数据矩阵，形状为 (样本数, 特征数)
    n_components (int): 降维后需要保留的维度数

    返回：
    eig_vectors (numpy.ndarray): 主成分特征向量
    eig_values (numpy.ndarray): 主成分特征值
    mean_data (numpy.ndarray): 数据均值
    projected_data (numpy.ndarray): 降维后的数据
    """
    # 数据中心化
    mean_data = np.mean(data, axis=0)
    centered_data = data - mean_data

    # 计算协方差矩阵
    if centered_data.shape[0] >= centered_data.shape[1]:
        cov_matrix = np.dot(centered_data.T, centered_data)
    else:
        cov_matrix = np.dot(centered_data, centered_data.T)

    # 计算特征值和特征向量
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

    # 排序特征值和特征向量
    sorted_indices = np.argsort(-eig_values)
    eig_values = eig_values[sorted_indices]
    eig_vectors = eig_vectors[:, sorted_indices]

    # 选择前 n_components 个主成分
    eig_values = eig_values[:n_components]
    eig_vectors = eig_vectors[:, :n_components]

    # 如果协方差矩阵是数据矩阵乘以自身的转置，需要转换特征向量
    if centered_data.shape[0] < centered_data.shape[1]:
        eig_vectors = np.dot(centered_data.T, eig_vectors)
        # 归一化特征向量
        for i in range(eig_vectors.shape[1]):
            eig_vectors[:, i] = eig_vectors[:, i] / np.linalg.norm(eig_vectors[:, i])

    # 降维后的数据
    projected_data = np.dot(centered_data, eig_vectors)

    return eig_vectors, eig_values, mean_data, projected_data

@timer
def perform_pca(Xtrain, n_components=None):
    """
    对训练数据进行 PCA 分析。

    参数：
    Xtrain (np.ndarray): 训练数据，形状为 (样本数, 特征数)。
    n_components (int, optional): 保留的主成分数量。

    返回：
    tuple: (eig_vectors, eig_values, mean_data, Xtr)
        - eig_vectors (np.ndarray): 主成分特征向量。
        - eig_values (np.ndarray): 主成分特征值。
        - mean_data (np.ndarray): 数据均值。
        - Xtr (np.ndarray): 降维后的训练数据。
    """
    if n_components is None:
        n_components = Xtrain.shape[1]
    eig_vectors, eig_values, mean_data, Xtr = PCA(Xtrain, n_components)
    return eig_vectors, eig_values, mean_data, Xtr

@timer
def knn(k, data_train, label_train, data_test):
    """
    KNN 分类器
    
    参数：
    k (int): 邻居数量
    data_train (np.ndarray): 训练数据，形状为 (样本数, 特征数)
    label_train (np.ndarray): 训练数据标签，形状为 (样本数,)
    data_test (np.ndarray): 测试数据，形状为 (样本数, 特征数)
    
    返回：
    label_test (np.ndarray): 测试数据的预测标签，形状为 (样本数,)
    """
    # 计算距离矩阵
    dist_matrix = np.linalg.norm(data_test[:, np.newaxis] - data_train, axis=2)
    
    # 对距离进行排序
    nearest_indices = np.argsort(dist_matrix, axis=1)[:, :k]
    
    # 获取最近邻的标签
    nearest_labels = label_train[nearest_indices]
    
    # 对最近邻的标签进行投票（对于 k=1，直接取最近邻的标签）
    if k == 1:
        label_test = nearest_labels.flatten()
    else:
        # 对每个测试样本，统计最近邻标签中出现次数最多的标签
        from scipy.stats import mode
        label_test, _ = mode(nearest_labels, axis=1)
        label_test = label_test.flatten()
    
    return label_test

@timer
def perform_knn_classification(knn_classifier, Xtr, train_label, Xts):
    """
    使用 KNeighborsClassifier 进行训练和预测。

    参数：
    knn_classifier (KNeighborsClassifier): 已初始化的 KNeighborsClassifier 实例。
    Xtr (np.ndarray): 训练数据，形状为 (样本数, 特征数)。
    train_label (np.ndarray): 训练数据标签，形状为 (样本数,)。
    Xts (np.ndarray): 测试数据，形状为 (样本数, 特征数)。

    返回：
    np.ndarray: 测试数据的预测标签，形状为 (样本数,)。
    """
    knn_classifier.fit(Xtr, train_label)
    testResults = knn_classifier.predict(Xts)
    return testResults

@timer
def perform_svm_classification(Xtr, train_label, Xts, classes, kernel_type='poly'):
    """
    使用 OneVsRestClassifier 和 SVC 进行 SVM 分类，并进行超参数调优。
    """
    # verbose = True
    # if verbose:
    #     print(f"Training SVM classifiers with kernel='{kernel_type}'...")
    
    # # 初始化 OneVsRestClassifier
    # svm = OneVsRestClassifier(SVC(kernel=kernel_type, probability=True, cache_size=1000))
    
    # # 定义超参数网格
    # param_grid = {
    #     'estimator__C': [0.1, 1, 10],
    #     'estimator__gamma': ['scale', 'auto'],
    #     'estimator__degree': [3, 4, 5] if kernel_type == 'poly' else [3]
    # }
    
    # # 使用 GridSearchCV 进行超参数调优
    # grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=1)
    # grid_search.fit(Xtr, train_label)
    
    # if verbose:
    #     print(f"Best parameters: {grid_search.best_params_}")
    #     print("Classifying test data with best SVM classifiers...")

    # 不使用超参数调优,将以上注释掉即可
    verbose = True
    if verbose:
        print(f"Training SVM classifiers with kernel='{kernel_type}'...")
    
    # 初始化 OneVsRestClassifier
    svm = OneVsRestClassifier(SVC(kernel=kernel_type, probability=True, cache_size=1000))
    
    # 训练模型
    svm.fit(Xtr, train_label)
    
    if verbose:
        print("Classifying test data with SVM classifiers...")
    
    # 预测
    test_results = svm.predict(Xts)
    
    return test_results

@timer
def perform_random_forest_classification(n_trees, X_train, train_label, X_test):
    """
    使用 RandomForestClassifier 进行随机森林分类。

    参数：
    n_trees (int): 随机森林中树的数量。
    X_train (np.ndarray): 训练数据，形状为 (样本数, 特征数)。
    train_label (np.ndarray): 训练数据标签，形状为 (样本数,)。
    X_test (np.ndarray): 测试数据，形状为 (样本数, 特征数)。

    返回：
    np.ndarray: 测试数据的预测标签，形状为 (样本数,)。
    """
    verbose = True
    if verbose:
        print(f"Training Random Forest with {n_trees} trees...")

    # 初始化 RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)

    # 训练模型
    rf.fit(X_train, train_label)

    if verbose:
        print("Classifying test data with Random Forest...")

    # 预测
    test_results = rf.predict(X_test)

    return test_results

@timer
def perform_deep_learning_classification(X_train, train_label, X_test, num_classes, run_output_path, epochs=100, batch_size=32):
    """
    使用深度学习（MLP）进行分类，并进行了多项优化。
    
    参数：
    X_train (np.ndarray): 训练数据，形状为 (样本数, 特征数)。
    train_label (np.ndarray): 训练数据标签，形状为 (样本数,)。
    X_test (np.ndarray): 测试数据，形状为 (样本数, 特征数)。
    num_classes (int): 类别数量。
    run_output_path (str): 运行文件夹路径，用于保存模型文件。
    epochs (int, optional): 训练轮数。默认值为 100。
    batch_size (int, optional): 批次大小。默认值为 32。
    
    返回：
    np.ndarray: 测试数据的预测标签，形状为 (样本数,)。
    """
    
    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 将标签转换为类别编码
    y_train = to_categorical(train_label - 1, num_classes)  # 假设标签从1开始
    
    # 计算类别权重（键从0开始）
    class_weights_array = class_weight.compute_class_weight('balanced',
                                                            classes=np.unique(train_label) - 1,
                                                            y=train_label - 1)
    class_weights = {i: class_weights_array[i] for i in range(num_classes)}
    
    # 构建更深且带有BatchNormalization和L2正则化的模型
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    # 编译模型，使用自定义学习率的Adam优化器
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # 定义早停、模型检查点和学习率调度
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint_path = os.path.join(run_output_path, 'best_model.h5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    # 训练模型，添加验证集
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2,
              callbacks=[early_stopping, checkpoint, reduce_lr],
              verbose=1,
              class_weight=class_weights)
    
    # 加载最佳模型
    model.load_weights(checkpoint_path)
            
    # 预测
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1) + 1  # 假设标签从1开始
    
    print(f"模型已保存至 {checkpoint_path}")
    
    return predicted_labels

def calculate_accuracy(outpca, labels, mask, class_num):
    """
    计算每一类的分类正确率和总体正确率。
    
    参数：
    outpca (np.ndarray): 模型的预测结果矩阵，形状为 (行数, 列数)
    labels (np.ndarray): 真实标签矩阵，形状为 (行数, 列数)
    mask (np.ndarray): 训练样本的掩码矩阵，形状为 (行数, 列数)
    class_num (int): 类别数量
    
    返回：
    accuracy (list): 每一类的分类正确率列表
    OA (float): 总体分类正确率
    """
    accuracy = []
    for i in range(1, class_num + 1):
        # 计算测试集（非训练集）的真实标签为当前类别的位置
        idx = (labels == i) & (mask == 0)
        # 防止分母为零
        total = np.sum(idx)
        if total == 0:
            acc = 0.0
        else:
            # 计算预测正确的数量
            correct = np.sum((outpca == i) & idx)
            acc = correct / total
        accuracy.append(acc)
    
    # 计算总体分类正确率
    test_idx = (labels > 0) & (mask == 0)
    OA = np.sum((outpca == labels) & test_idx) / np.sum(test_idx)
    
    return accuracy, OA

def set_fig_zhcn():
    """
    设置图表为中文显示。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

def keep_figures_open():
    try:
        while plt.get_fignums():
            plt.pause(100)
    except KeyboardInterrupt:
        plt.close('all')  # 关闭所有画布

@log_prints
def main(run_output_path):
    """
    - 运行多种分类方法（优化KNN、KNN、RF、SVM、深度学习）
    - 存储和比较每种方法的分类结果
    """
    set_fig_zhcn()
    
    mat_file_path = r"D:\Users\admin\Documents\MATLAB\moshishibie_lib\上课实验代码\机器学习特征提取与分类\UPavia.mat"
    
    # 加载数据并预处理
    pavia, rgb_pavia, test_pavia, X = load_and_preprocess_data(mat_file_path)

    # 统计类别数
    class_num, T = calculate_class_statistics(test_pavia)

    # 绘制 RGB 影像
    plot_rgb_image(rgb_pavia, run_output_path)

    # 选择训练样本的数量
    numsample = 100  # 每类样本选择 100 个
    trainlabels, testlabels = getlabeled(test_pavia, numsample)

    print('总体训练样本的数量为:')
    Num_train = np.sum(trainlabels != 0)
    print(Num_train)

    # 提取训练样本
    xxt, yyt = np.where(trainlabels != 0)
    Xtrain = pavia[xxt, yyt, :]
    train_label = trainlabels[xxt, yyt]

    # 将训练样本重塑为二维数组
    Xtrain = Xtrain.reshape(-1, pavia.shape[2])

    # 对训练数据进行 PCA 分析，不截断特征值和特征向量
    eig_vectors_full, eig_values_full, mean_data, _ = perform_pca(Xtrain)

    # 绘制累计解释方差曲线
    cumulative_explained_variance = plot_cumulative_explained_variance(eig_values_full, run_output_path)

    # 设置累计解释方差阈值，根据阈值选择主成分数量
    threshold = 0.9999  # 累计解释方差阈值
    n_components = np.argmax(cumulative_explained_variance >= threshold) + 1
    print(f"选择前 {n_components} 个主成分可达到 {round(threshold * 100, 4)}% 的累计解释方差。")

    # 使用选定的 n_components 重新进行 PCA 降维
    eig_vectors, eig_values, mean_data, Xtr = perform_pca(Xtrain, n_components)
    
    # 对所有数据进行 PCA 投影
    centered_X = X - mean_data
    Xts = np.dot(centered_X, eig_vectors)

    # 定义分类方法
    classifiers = {
        "优化KNN": {
            "type": "knn",
            "params": {"k": 1, "optimized": True}  # 优化后的KNN
        },
        # "KNN": {
        #     "type": "knn",
        #     "params": {"k": 1, "optimized": False}  # 非优化的KNN
        # },
        "随机森林": {
            "type": "rf",
            "params": {"n_trees": 100}
        },
        "SVM": {
            "type": "svm",
            "params": {"kernel_type": "poly"}
        },
        "深度学习": {
            "type": "dl",
            "params": {"num_classes": class_num, "epochs": 100, "batch_size": 32}
        }
    }

    results = {}

    for name, classifier in classifiers.items():
        print(f"\n=== 运行分类方法：{name} ===")
        if classifier["type"] == "knn":
            if classifier["params"]["optimized"]:
                # 优化后的KNN（使用scikit-learn的KNeighborsClassifier）
                knn_classifier = KNeighborsClassifier(
                    n_neighbors=classifier["params"]["k"], 
                    algorithm='auto', 
                    n_jobs=-1
                )
                testResults = perform_knn_classification(
                    knn_classifier, 
                    Xtr, 
                    train_label, 
                    Xts
                )
            else:
                # 非优化的KNN（使用自定义的knn函数）
                testResults = knn(
                    k=classifier["params"]["k"], 
                    data_train=Xtr, 
                    label_train=train_label, 
                    data_test=Xts
                )
        
        elif classifier["type"] == "rf":
            testResults = perform_random_forest_classification(
                n_trees=classifier["params"]["n_trees"], 
                X_train=Xtr, 
                train_label=train_label, 
                X_test=Xts
            )
        
        elif classifier["type"] == "svm":
            classes = np.unique(train_label)
            testResults = perform_svm_classification(
                Xtr, 
                train_label, 
                Xts, 
                classes, 
                kernel_type=classifier["params"]["kernel_type"]
            )
        
        elif classifier["type"] == "dl":
            testResults = perform_deep_learning_classification(
                X_train=Xtr,
                train_label=train_label,
                X_test=Xts,
                num_classes=classifier["params"]["num_classes"],
                run_output_path=run_output_path,  # 传递 run_output_path
                epochs=classifier["params"]["epochs"],
                batch_size=classifier["params"]["batch_size"]
            )
        
        # 将预测结果重塑为原始图像的形状
        outpca = testResults.reshape(pavia.shape[0], pavia.shape[1])
        
        # 计算分类精度
        mask = trainlabels
        labels = test_pavia
        class_num_current = len(np.unique(labels[labels > 0]))
        accuracy, OA = calculate_accuracy(outpca, labels, mask, class_num_current)
        
        # 存储结果
        results[name] = {
            "accuracy": accuracy,
            "OA": OA,
            "outpca_display": outpca.copy()
        }
        
        # 打印每一类的分类正确率
        print('每一类的分类正确率为：')
        for i, acc in enumerate(accuracy, 1):
            print(f"第 {i} 类正确率为： {acc * 100:.2f}%")
        
        print(f'总体分类正确率为: {OA * 100:.2f}%')
        
        # 显示分类结果，只显示测试样本
        outpca_display = outpca.copy()
        outpca_display[mask != 0] = 0  # 将训练样本位置置为 0，不显示
        plt.figure()
        plt.imshow(outpca_display, cmap='jet')
        plt.title(f'{name} 分类结果')
        plt.axis('off')
        plt.show()
        
        # 保存分类结果
        OA_percentage = OA * 100
        filename = os.path.join(run_output_path, f'Upavia_{name}_{OA_percentage:.2f}_.png')
        plt.imsave(filename, outpca_display, cmap='jet')
    
    # 比较五种方法的分类结果
    print("\n=== 分类方法比较 ===")
    for name, res in results.items():
        print(f"\n分类方法：{name}")
        print(f"总体分类正确率（OA）: {res['OA'] * 100:.2f}%")
        for i, acc in enumerate(res["accuracy"], 1):
            print(f"第 {i} 类正确率： {acc * 100:.2f}%")
    
    # 简要分析结果（示例）
    print("\n=== 简要分析 ===")
    best_method = max(results.items(), key=lambda x: x[1]["OA"])
    print(f"最佳分类方法是：{best_method[0]}，总体分类正确率为 {best_method[1]['OA'] * 100:.2f}%")
    
    # 绘制验证集的正确结果
    plot_validation_results(test_pavia, 
                            title="验证集正确结果", 
                            filepath=os.path.join(
                                run_output_path, 
                                'Validation_Truth.png'
                            )
                        )

    keep_figures_open()

if __name__ == "__main__":
    # 指定输出路径
    base_output_path = r"D:\Users\admin\Documents\MATLAB\moshishibie_lib\第四次作业-高光谱降维与分类\outputs"
    
    # 创建时间戳文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_path = os.path.join(base_output_path, f"run_{timestamp}")
    os.makedirs(run_output_path, exist_ok=True)
    
    print(f"所有输出文件将保存在: {run_output_path}")  # 添加此行用于验证路径
    
    # 调用 main 函数并传递 run_output_path
    main(run_output_path=run_output_path)