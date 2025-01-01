import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
import os

def get_manual_results():
    """
    返回方法的手动输入结果（混淆矩阵、分类报告、R2、MSE、RMSE等）。
    """
    data = {
        "RandomForest": {
            "cm": np.array([
                [115,126, 36,  2,  1,  0],
                [ 36,499, 53, 24,  0,  0],
                [ 10, 35,126, 11,  0,  1],
                [  8, 71, 25,  9,  0,  0],
                [  3,  1,  4,  0,  0,  0],
                [  0,  0,  0,  0,  0,625]
            ]),
            "report": """              precision    recall  f1-score   support

           0       0.67      0.41      0.51       280
           1       0.68      0.82      0.74       612
           2       0.52      0.69      0.59       183
           3       0.20      0.08      0.11       113
           4       0.00      0.00      0.00         8
           5       1.00      1.00      1.00       625

        accuracy                           0.75      1821
    macro avg       0.51      0.50      0.49      1821
    weighted avg       0.74      0.75      0.74      1821
    """,
            "r2": 0.854251680826419,
            "mse": 0.5700164744645799,
            "rmse": 0.7549943539289415
        },
        "KNN": {
            "cm": np.array([
                [ 63, 97, 28, 36, 10, 46],
                [133,245, 80, 77, 13, 64],
                [ 30, 47, 33, 17, 11, 45],
                [ 16, 39, 17, 21,  4, 16],
                [  1,  0,  3,  1,  1,  2],
                [ 72, 95, 69, 43, 29,317]
            ]),
            "report": """              precision    recall  f1-score   support

           0       0.20      0.23      0.21       280
           1       0.47      0.40      0.43       612
           2       0.14      0.18      0.16       183
           3       0.11      0.19      0.14       113
           4       0.01      0.12      0.03         8
           5       0.65      0.51      0.57       625

        accuracy                           0.37      1821
    macro avg       0.26      0.27      0.26      1821
    weighted avg       0.43      0.37      0.40      1821
    """,
            "r2": -0.2210283078357036,
            "mse": 4.775398132894014,
            "rmse": 2.1852684349740685
        },
        "SVM-rbf": {
            "cm": np.array([
                [  6,163, 21, 11, 24, 55],
                [  6,420, 22, 39, 18,107],
                [  4, 70, 30, 18, 21, 40],
                [  0, 70,  8,  4,  9, 22],
                [  0,  2,  1,  0,  3,  2],
                [  3,243, 23, 20, 49,287]
            ]),
            "report": """              precision    recall  f1-score   support

           0       0.32      0.02      0.04       280
           1       0.43      0.69      0.53       612
           2       0.29      0.16      0.21       183
           3       0.04      0.04      0.04       113
           4       0.02      0.38      0.05         8
           5       0.56      0.46      0.50       625

        accuracy                           0.41      1821
    macro avg       0.28      0.29      0.23      1821
    weighted avg       0.42      0.41      0.38      1821
    """,
            "r2": -0.3225466457571864,
            "mse": 5.172432729269632,
            "rmse": 2.2742982938193554
        },
        "SVM-linear": {
            "cm": np.array([
                [111,  98,  24,   6,  20,  21],
                [ 72, 443,  44,  35,  14,   4],
                [ 24,  33,  85,  12,  19,  10],
                [ 13,  54,  26,  12,   4,   4],
                [  3,   0,   1,   0,   4,   0],
                [  2,   0,   0,   0,   3, 620]
            ]),
            "report": """              precision    recall  f1-score   support

            0       0.49      0.40      0.44       280
            1       0.71      0.72      0.71       612
            2       0.47      0.46      0.47       183
            3       0.18      0.11      0.13       113
            4       0.06      0.50      0.11         8
            5       0.94      0.99      0.97       625

        accuracy                           0.70      1821
    macro avg       0.48      0.53      0.47      1821
    weighted avg       0.70      0.70      0.70      1821
    """,
            "r2": 0.6728385513733683,
            "mse": 1.2795167490389896,
            "rmse": 1.1311572609672758
        },
        "DecisionTree": {
            "cm": np.array([
                [133,  80,  36,  24,   7,   0],
                [112, 375,  58,  65,   2,   0],
                [ 25,  49,  85,  20,   3,   1],
                [ 20,  56,  24,  13,   0,   0],
                [  6,   1,   1,   0,   0,   0],
                [  0,   0,   0,   0,   0, 625]
            ]),
            "report": """              precision    recall  f1-score   support

            0       0.45      0.47      0.46       280
            1       0.67      0.61      0.64       612
            2       0.42      0.46      0.44       183
            3       0.11      0.12      0.11       113
            4       0.00      0.00      0.00         8
            5       1.00      1.00      1.00       625

        accuracy                           0.68      1821
    macro avg       0.44      0.44      0.44      1821
    weighted avg       0.68      0.68      0.68      1821
    """,
            "r2": 0.757507372627385,
            "mse": 0.9483800109829764,
            "rmse": 0.9738480430657426
        },
        "GradientBoosting": {
            "cm": np.array([
                [137, 117,  23,   3,   0,   0],
                [ 29, 536,  30,  16,   1,   0],
                [ 18,  58,  98,   7,   1,   1],
                [  8,  79,  17,   9,   0,   0],
                [  4,   1,   3,   0,   0,   0],
                [  0,   0,   0,   0,   0, 625]
            ]),
            "report": """              precision    recall  f1-score   support

            0       0.70      0.49      0.58       280
            1       0.68      0.88      0.76       612
            2       0.57      0.54      0.55       183
            3       0.26      0.08      0.12       113
            4       0.00      0.00      0.00         8
            5       1.00      1.00      1.00       625

        accuracy                           0.77      1821
    macro avg       0.53      0.50      0.50      1821
    weighted avg       0.75      0.77      0.75      1821
    """,
            "r2": 0.8584640599932856,
            "mse": 0.5535420098846787,
            "rmse": 0.744004038890031
        },
        "LogisticRegression": {
            "cm": np.array([
                [119,  91,  27,  22,  12,   9],
                [ 86, 406,  44,  70,   6,   0],
                [ 22,  33,  98,  24,   5,   1],
                [ 14,  47,  13,  36,   3,   0],
                [  1,   1,   1,   1,   4,   0],
                [  3,   0,   1,   1,   1, 619]
            ]),
            "report": """              precision    recall  f1-score   support

            0       0.49      0.42      0.45       280
            1       0.70      0.66      0.68       612
            2       0.53      0.54      0.53       183
            3       0.23      0.32      0.27       113
            4       0.13      0.50      0.21         8
            5       0.98      0.99      0.99       625

        accuracy                           0.70      1821
    macro avg       0.51      0.57      0.52      1821
    weighted avg       0.72      0.70      0.71      1821
    """,
            "r2": 0.7330755734595593,
            "mse": 1.0439319055464031,
            "rmse": 1.0217298593788884
        },
        "GaussianNB": {
            "cm": np.array([
                [ 16, 216,  20,  12,  15,   1],
                [  8, 533,  18,  40,  13,   0],
                [ 18, 102,  27,   9,  26,   1],
                [  3,  84,  11,  10,   4,   1],
                [  1,   3,   0,   0,   4,   0],
                [  1, 136,  24,   6,  17, 441]
            ]),
            "report": """              precision    recall  f1-score   support

            0       0.34      0.06      0.10       280
            1       0.50      0.87      0.63       612
            2       0.27      0.15      0.19       183
            3       0.13      0.09      0.11       113
            4       0.05      0.50      0.09         8
            5       0.99      0.71      0.83       625

        accuracy                           0.57      1821
    macro avg       0.38      0.39      0.32      1821
    weighted avg       0.60      0.57      0.54      1821
    """,
            "r2": 0.41714713594457153,
            "mse": 2.2795167490389896,
            "rmse": 1.5098068581904738
        },
        "BernoulliNB": {
            "cm": np.array([
                [ 34,  24,  22,   7,  53, 140],
                [ 34, 112,  64,  30,  54, 318],
                [ 10,   6,  20,   1,  30, 116],
                [  3,   8,  23,   4,  13,  62],
                [  0,   0,   1,   0,   0,   7],
                [ 23,   3,  21,  10,  54, 514]
            ]),
            "report": """              precision    recall  f1-score   support

            0       0.33      0.12      0.18       280
            1       0.73      0.18      0.29       612
            2       0.13      0.11      0.12       183
            3       0.08      0.04      0.05       113
            4       0.00      0.00      0.00         8
            5       0.44      0.82      0.58       625

        accuracy                           0.38      1821
    macro avg       0.29      0.21      0.20      1821
    weighted avg       0.47      0.38      0.34      1821
    """,
            "r2": -0.7953160009185036,
            "mse": 7.021416803953872,
            "rmse": 2.6497956155058207
        },
    }
    return data

def compute_label_metrics_single_method(conf_mat, method_report):
    """
    从混淆矩阵和分类报告中解析每个标签的相关统计信息
    返回一个列表，元素为：{ 'label': i, 'precision': ..., 'recall': ..., 'f1': ..., '漏分': ..., '错分': ... }
    """
    lines = method_report.split('\n')
    label_metrics = []
    # 利用报告表格解析
    for line in lines:
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit():
            i = int(parts[0])
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            # support = int(parts[4]) # 用不到也可以保留
            # 计算“漏分”和“错分”示例(自己定义，这里仅简单展示)
            row_sum = np.sum(conf_mat[i])
            col_sum = np.sum(conf_mat[:, i])
            漏分 = row_sum - conf_mat[i, i]
            错分 = col_sum - conf_mat[i, i]
            label_metrics.append({
                'label': i,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                '漏分': 漏分,
                '错分': 错分
            })
    return label_metrics

def compute_overall_metrics_single_method(conf_mat, method_report, r2, mse, rmse):
    """
    计算总体准确率、kappa、r2、mse、rmse
    返回一个dict
    """
    # 总体准确率
    correct = np.trace(conf_mat)
    total = np.sum(conf_mat)
    acc = correct / total
    # kappa
    y_true = []
    y_pred = []
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            y_true.extend([i]*conf_mat[i,j])
            y_pred.extend([j]*conf_mat[i,j])
    kappa = cohen_kappa_score(y_true, y_pred)
    return {
        'accuracy': acc,
        'kappa': kappa,
        'r2': r2,
        'mse': mse,
        'rmse': rmse
    }

def plot_label_metrics_bar(label_metrics_all_methods, label_index=0, metric='precision', output_dir='figures'):
    """
    根据给定标签和指标绘制柱状图并保存到指定位置
    label_metrics_all_methods: { method_name: [{label:0,precision,recall,f1,漏分,错分}, ...], ... }
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 收集每个方法该标签的指标
    method_vals = []
    for method, lm in label_metrics_all_methods.items():
        for item in lm:
            if item['label'] == label_index:
                method_vals.append((method, item[metric]))
                break
    # 按值排序
    method_vals.sort(key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(8,6))
    methods = [x[0] for x in method_vals]
    vals = [x[1] for x in method_vals]
    bars = plt.bar(methods, vals)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, height*1.01, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    plt.ylabel(metric)
    plt.title(f'标签 {label_index} 的 {metric} 对比')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # 保存图像
    save_path = os.path.join(output_dir, f'label_{label_index}_{metric}.png')
    plt.savefig(save_path)
    plt.close()

    # 打印方法排序及其指标
    sorted_methods = ' > '.join([f'{method}({val:.2f})' for method, val in method_vals])
    print(f'标签 {label_index} 的 {metric} 排序: {sorted_methods}')

def plot_overall_metrics_bar(overall_metrics_all_methods, sort_metric='accuracy', output_dir='figures'):
    """
    绘制总体指标的柱状图，对比accuracy, kappa, r2, mse, rmse等，并保存到指定位置
    overall_metrics_all_methods: {method: {accuracy, kappa, r2, mse, rmse}}
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按照sort_metric排序
    sorted_items = sorted(overall_metrics_all_methods.items(), key=lambda x: x[1][sort_metric], reverse=True)
    methods = [x[0] for x in sorted_items]
    # 准备需要展示的指标
    metrics_list = ['accuracy','kappa','r2','mse','rmse']
    for mkey in metrics_list:
        plt.figure(figsize=(8,6))
        vals = [overall_metrics_all_methods[mt][mkey] for mt in methods]
        bars = plt.bar(methods, vals)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, height*1.01, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        plt.ylabel(mkey)
        plt.title(f'总体 {mkey} 对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # 保存图像
        save_path = os.path.join(output_dir, f'overall_{mkey}.png')
        plt.savefig(save_path)
        plt.close()

        # 打印方法排序及其指标
        sorted_methods = ' > '.join([f'{method}({val:.2f})' for method, val in zip(methods, vals)])
        print(f'总体 {mkey} 排序: {sorted_methods}')

def analyze_and_plot_all(data, output_dir='figures'):
    """
    主函数：计算并绘制所有方法的标签级指标和总体指标，并保存到指定位置
    """
    # 1. 逐方法计算标签级指标
    label_metrics_all_methods = {}
    overall_metrics_all_methods = {}
    for method, info in data.items():
        cm = info['cm']
        rep = info['report']
        r2 = info['r2']
        mse = info['mse']
        rmse = info['rmse']
        label_metrics_all_methods[method] = compute_label_metrics_single_method(cm, rep)
        overall_metrics_all_methods[method] = compute_overall_metrics_single_method(cm, rep, r2, mse, rmse)
    # 2. 示例：绘制标签0的precision排名柱状图
    # plot_label_metrics_bar(label_metrics_all_methods, label_index=0, metric='precision', output_dir=output_dir)
    for _ in range(0,6):
        for metric in ['precision','recall','f1','漏分','错分']:
            plot_label_metrics_bar(label_metrics_all_methods, label_index=_, metric=metric, output_dir=output_dir)
    # 3. 绘制总体指标排名柱状图
    plot_overall_metrics_bar(overall_metrics_all_methods, sort_metric='accuracy', output_dir=output_dir)
if __name__ == '__main__':
    data = get_manual_results()
    analyze_and_plot_all(data,r'd:\Users\admin\Documents\MATLAB\moshishibie_lib\期末课程设计\data\plots\restlts')