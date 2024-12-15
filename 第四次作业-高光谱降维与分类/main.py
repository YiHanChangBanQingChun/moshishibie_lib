import scipy.io
import numpy as np
import matplotlib.pyplot as plt

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

def get_test_pavia_class_count(mat_file_path):
    """
    读取 .mat 文件中的 Test_Pavia 数据，并返回其类别数。

    参数：
    mat_file_path (str): .mat 文件的路径。

    返回：
    int: Test_Pavia 的类别数。
    """
    # 获取 Test_Pavia 数据
    test_pavia_info = get_mat_variable_info(mat_file_path, 'Test_Pavia')
    test_pavia_data = test_pavia_info['data']
    
    # 计算类别数
    unique_classes = np.unique(test_pavia_data)
    class_count = len(unique_classes)
    
    return class_count

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

def set_fig_zhcn():
    """
    设置图表为中文显示。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

def main():
    """
    数据分为四部分：
    - 总数据：Pavia：shape=(610, 340, 103), dtype=int16
    - RGB显示：RGB_Pavia：shape=(610, 340, 3), dtype=uint8
    - 用于训练：Train_Pavia：shape=(610, 340), dtype=uint8
    - 用于测试：Test_Pavia：shape=(610, 340), dtype=uint8
    """

    set_fig_zhcn()

    mat_file_path = r"D:\Users\admin\Documents\MATLAB\moshishibie_lib\上课实验代码\机器学习特征提取与分类\UPavia.mat"
    
    # 加载数据
    data_info = get_mat_variable_info(mat_file_path)
    pavia = data_info['Pavia']['data'].astype(np.float64)
    rgb_pavia = data_info['RGB_Pavia']['data']
    test_pavia = data_info['Test_Pavia']['data']
    train_pavia = data_info['Train_Pavia']['data']

    # 将数据重塑为二维数组（像素数 x 波段数）
    X = pavia.reshape(-1, pavia.shape[2])

    # 统计类别数
    numlabeled = test_pavia[test_pavia > 0]
    numN = np.unique(numlabeled)
    num = len(numN)
    T = []
    N = []
    for i in numN:
        N_i = np.sum(test_pavia == i)
        T.append(N_i)
    class_num = num

    # 显示 RGB 影像
    plt.imshow(rgb_pavia)
    plt.title('RGB_Pavia 影像')
    plt.show()

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

if __name__ == "__main__":
    main()