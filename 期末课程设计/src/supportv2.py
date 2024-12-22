import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import json
import time
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from functools import wraps
import sys
import os
from sklearn.svm import LinearSVC

def redirect_output(file_path):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            with open(file_path, 'a', encoding='utf-8') as f:
                sys.stdout = f
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    print("程序被手动终止")
                    raise
                finally:
                    sys.stdout = original_stdout
        return wrapper
    return decorator

# 运行时间打印装饰器
def print_run_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 运行时间：{end_time - start_time:.2f} 秒")
        return result
    return wrapper

@dataclass
class Patient:
    """
    患者数据类
    """
    id: int  # 患者ID
    age: float  # 年龄，单位：岁
    death: float  # 死亡时间
    sex: int  # 性别，{male: 0, female: 1}
    hospdead: int  # 是否在医院死亡，{0: False, 1: True}
    slos: float  # 从研究进入到出院的天数
    d_time: float  # 随访天数
    dzgroup: int  # 疾病子类别，具体映射见下方
    dzclass: int  # 疾病类别，{ARF/MOSF: 0, COPD/CHF/Cirrhosis: 1, Cancer: 2, Coma: 3}
    num_co: int  # 同时存在的疾病数量
    edu: int  # 教育程度，年份
    income: int  # 收入水平，{"$11-$25k": 0, "$25-$50k": 1, ">$50k": 2, "under $11k": 3}
    scoma: float  # SUPPORT 第3天昏迷评分基于格拉斯哥量表（由模型预测）
    charges: float  # 医院费用
    totcst: float  # 成本与费用的总比例（RCC）成本
    totmcst: float  # 总微观成本
    avtisst: float  # 平均TISS得分，第3-25天
    race: int  # 种族，{asian: 0, black: 1, hispanic: 2, missing: 3, other: 4, white: 5}
    sps: float  # SUPPORT 第3天生理评分（由模型预测）
    aps: float  # APACHE III 第3天生理评分（无昏迷，imp bun, out for ph1）
    surv2m: float  # SUPPORT 模型第3天2个月存活估计（由模型预测）
    surv6m: float  # SUPPORT 模型第3天6个月存活估计（由模型预测）
    hday: int  # 患者进入研究时在医院的天数
    diabetes: int  # 是否患有糖尿病（1表示是，0表示否）
    dementia: int  # 是否患有痴呆症（1表示是，0表示否）
    ca: int  # 是否患有癌症，{no: 0, yes: 1, metastatic: 2}
    prg2m: float  # 医生对患者2个月存活的估计
    prg6m: int  # 医生对患者6个月存活的估计，{no: 0, yes: 1}
    dnr: int  # 是否有不施行抢救指令，{no dnr: 0, dnr after sadm: 1, dnr before sadm: 2, missing: 3}
    dnrday: float  # 不施行抢救指令的天数（<0表示研究前）
    meanbp: float  # 第3天的平均动脉血压
    wblc: float  # 第3天的白细胞计数（千计）
    hrt: float  # 第3天的心率
    resp: float  # 第3天的呼吸频率
    temp: float  # 第3天的体温（摄氏度）
    pafi: float  # 第3天的$PaO_2/FiO_2$比率
    alb: float  # 第3天的血清白蛋白水平
    bili: float  # 第3天的血清胆红素水平
    crea: float  # 第3天的血清肌酐水平
    sod: float  # 第3天的血清钠浓度
    ph: float  # 血液pH值
    glucose: int  # 第3天的血糖水平
    bun: int  # 第3天的血尿素氮水平
    urine: int  # 第3天的尿量
    adlp: int  # 第3天患者填写的日常生活活动指数，具体映射见下方
    adls: float  # 第3天由替代者填写的日常生活活动指数
    sfdm2: int  # 功能性残疾水平，1-5评分，具体映射见下方
    adlsc: float  # 校准后的日常生活活动指数

# 映射字典
SEX_MAPPING = {
    'male': 0, 
    'female': 1
    }
INCOME_MAPPING = {
    "_11-_25k": 0, 
    "_25-_50k": 1, 
    ">_50k": 2, 
    "under__11k": 3
    }
RACE_MAPPING = {
    'asian': 0, 
    'black': 1, 
    'hispanic': 2, 
    'missing': 3, 
    'other': 4, 
    'white': 5
    }
CA_MAPPING = {
    'no': 0, 
    'yes': 1, 
    'metastatic': 2
    }
DNR_MAPPING = {
    'no_dnr': 0, 
    'dnr_after_sadm': 1, 
    'dnr_before_sadm': 2, 
    }
DZGROUP_MAPPING = {
    'arf_mosf_w_sepsis': 0,
    'chf': 1,
    'copd': 2,
    'cirrhosis': 3,
    'colon_cancer': 4,
    'coma': 5,
    'lung_cancer': 6,
    'mosf_w_malig': 7
    }
DZCLASS_MAPPING = {
    'arf_mosf': 0,
    'copd_chf_cirrhosis': 1,
    'cancer': 2,
    'coma': 3
    }
SFDM2_MAPPING = {
    "no(M2 and SIP pres)": 1,
    "adl>=4 (>=5 if sur)": 2,
    "SIP>=30": 3,
    "Coma or Intub": 4,
    "<2 mo. follow-up": 5,
    }

def apply_mapping(df, column, mapping_dict):
    if df[column].dtype == 'object' and column != 'sfdm2':
        df[column] = df[column].astype(str).str.lower()
        df[column] = (
            df[column]
            .str.replace('/', '_')   # 新增处理，使 “/” 转为 "_"
            .str.replace(' ', '_')  
            .str.replace(r'[()\.\']', '', regex=True)
            .str.replace('$', '_', regex=False)
        )
        df[column] = df[column].map(mapping_dict).fillna(-1).astype(int)
    else:
        df[column] = df[column].map(mapping_dict).fillna(-1).astype(int)
    return df

def map_sfdm2(val):
    if pd.isnull(val):
        return 0
    if isinstance(val, (int, float)):
        if val in [1, 2, 3, 4, 5]:
            return int(val)
        else:
            return 0
    elif isinstance(val, str):
        val = val.strip()
        return SFDM2_MAPPING.get(val, 0)
    else:
        return 0

def preprocess_data(file_path):
    df = pd.read_csv(f'{file_path}/support2_full.csv')

    columns_to_map = [
        ('sex', SEX_MAPPING),
        ('dzgroup', DZGROUP_MAPPING),
        ('dzclass', DZCLASS_MAPPING),
        ('income', INCOME_MAPPING),
        ('race', RACE_MAPPING),
        ('ca', CA_MAPPING),
        ('dnr', DNR_MAPPING),
    ]
    for col, mapping_dict in columns_to_map:
        df = apply_mapping(df, col, mapping_dict)

    for col, _ in columns_to_map:
        missing_count = df[col].isnull().sum()
        print(f"缺失的'{col}'数量: {missing_count}")
        print(df[col].value_counts())

    df['sfdm2'] = df['sfdm2'].apply(map_sfdm2)
    missing_count = df['sfdm2'].isnull().sum()
    print(f"缺失的'sfdm2'数量: {missing_count}")
    print(df['sfdm2'].value_counts())

    df.fillna(-1, inplace=True)
    print("缺失值已填-1")

    df.to_csv(f'{file_path}/support2_full_preprocessed.csv', index=False)
    print("预处理后的数据已保存为 support2_full_preprocessed.csv")

def plot_stacked_bar_all_cols(file_path):
    df = pd.read_csv(f'{file_path}/support2_full_preprocessed.csv')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    def categorize_column(series, bins=10):
        """将连续变量分箱"""
        if series.nunique() > bins:
            categorized = pd.cut(series[series != -1], bins=bins, labels=False)
            categorized = pd.Series(categorized, index=series[series != -1].index)
            categorized = categorized.reindex(series.index, fill_value=-1)
            return categorized
        return series

    cat_counts = pd.DataFrame()
    for col in df.columns:
        categorized_col = categorize_column(df[col])
        col_counts = categorized_col.value_counts(normalize=True).sort_index()
        col_counts.name = col
        cat_counts = pd.concat([cat_counts, col_counts], axis=1)

    cat_counts.fillna(0, inplace=True)
    
    # 生成足够的颜色
    num_colors = len(cat_counts)
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
    
    # 按索引排序
    sorted_cat_counts = cat_counts.T.sort_index(axis=1)
    
    sorted_cat_counts.plot(kind='bar', stacked=True, figsize=(20, 8), color=colors)
    plt.title("所有列的堆积百分比分布")
    plt.xlabel("列名")
    plt.ylabel("百分比")
    plt.legend(sorted_cat_counts.columns, bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.show()
    plt.savefig(f"{file_path}/stacked_bar_all_cols.png")

def download_data(file_path):
    # 下载数据集
    url = 'https://archive.ics.uci.edu/static/public/880/data.csv'
    df = pd.read_csv(url)

    # 保存为CSV文件
    df.to_csv(f'{file_path}/support2_full.csv', index=False)

    print("数据已保存为 support2_full.csv")

def preprocess_for_naive_bayes(X):
    """
    预处理数据使其适合朴素贝叶斯分类器:
    1. 将负值替换为0
    2. 对数据进行Min-Max缩放到[0,1]区间
    """
    # 将负值替换为0
    X = np.where(X < 0, 6, X)
    
    # Min-Max缩放
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # 处理可能的nan值
    X = np.nan_to_num(X, 0)
    
    return X

def train_and_evaluate(file_path, model, param_grid=None, classifier_name=None, test_size=0.2, do_preproces = False):

    df = pd.read_csv(f'{file_path}/support2_full_preprocessed.csv')

    # 移除id列、目标列sfdm2以及不参与预测的列death和hospdead
    X = df.drop(columns=['id', 'sfdm2', 'death', 'hospdead'])
    y = df['sfdm2']

    # 分割数据集，80%训练，20%验证
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # 优化方向1，处理不平衡数据---
    # 结合欠采样和过采样
    over = SMOTE(random_state=42)
    under = RandomUnderSampler(random_state=42)
    X_train, y_train = over.fit_resample(X_train, y_train)
    X_train, y_train = under.fit_resample(X_train, y_train)

    # 定义预处理步骤
    if do_preproces:
        categorical_features = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'dnr']
        continuous_features = [col for col in X.columns if col not in categorical_features]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), continuous_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # 创建管道
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', model)])
        
    # 如果是朴素贝叶斯分类器，进行特殊预处理
    if isinstance(model, (MultinomialNB, BernoulliNB)):
        X_train = preprocess_for_naive_bayes(X_train.values)
        X_val = preprocess_for_naive_bayes(X_val.values)

    # 如果提供了参数网格，使用网格搜索法
    if param_grid:
        start_time = time.time()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        end_time = time.time()

        best_params = grid_search.best_params_
        processing_time = end_time - start_time
        print(f"{classifier_name} 最佳参数：{best_params}")
        print(f"{classifier_name} 格网搜索处理时间：{processing_time:.2f} 秒")
        with open(f"{file_path}/{classifier_name}_best_params.json", "w") as f:
            json.dump(best_params, f)

        # 训练模型
    model.fit(X_train, y_train)

    # 如果是随机森林，提取特征重要性
    if isinstance(model, RandomForestClassifier):
        feature_importances = model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # 打印特征重要性
        print("特征重要性：")
        print(feature_importance_df)

        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('随机森林特征重要性')
        # plt.show()
        plt.savefig(f"{file_path}/random_forest_feature_importance.png")

    # 预测验证集
    y_val_pred = model.predict(X_val)

    # 计算混淆矩阵
    cm = confusion_matrix(y_val, y_val_pred)
    print("验证集混淆矩阵：")
    print(cm)

    # 计算分类报告
    cr = classification_report(y_val, y_val_pred, zero_division=0)
    print("验证集分类报告：")
    print(cr)

    # 计算R2得分
    r2 = r2_score(y_val, y_val_pred)
    print("验证集R2得分：", r2)

    # 计算MSE和RMSE
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    print("验证集MSE：", mse)
    print("验证集RMSE：", rmse)
    # 将预测结果添加到 CSV 文件中
    df['predicted_sfdm2'] = model.predict(X)
    df.to_csv(f'{file_path}/support2_full_preprocessed_with_predictions.csv', index=False)

    # 返回验证集的预测结果和真实值
    return y_val, y_val_pred, cm, cr, r2

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title(title)
    # plt.show()
    plt.savefig(f"{title}.png")

def plot_pca_variance(file_path):
    df = pd.read_csv(f'{file_path}/support2_full_preprocessed.csv')

    # 移除id列和目标列sfdm2
    X = df.drop(columns=['id', 'sfdm2', 'death', 'hospdead'])

    # 标准化数据
    X = (X - X.mean()) / X.std()

    # 进行PCA
    pca = PCA()
    pca.fit(X)

    # 绘制累计解释方差曲线
    plt.figure(figsize=(10, 7))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差')
    plt.title('累计解释方差曲线')
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{file_path}/pca_variance.png")

    # 返回PCA对象
    return pca

def preprocess_data_with_pca(file_path, n_components):
    df = pd.read_csv(f'{file_path}/support2_full.csv')

    columns_to_map = [
        ('sex', SEX_MAPPING),
        ('dzgroup', DZGROUP_MAPPING),
        ('dzclass', DZCLASS_MAPPING),
        ('income', INCOME_MAPPING),
        ('race', RACE_MAPPING),
        ('ca', CA_MAPPING),
        ('dnr', DNR_MAPPING),
    ]
    for col, mapping_dict in columns_to_map:
        df = apply_mapping(df, col, mapping_dict)

    for col, _ in columns_to_map:
        missing_count = df[col].isnull().sum()
        print(f"缺失的'{col}'数量: {missing_count}")
        print(df[col].value_counts())

    df['sfdm2'] = df['sfdm2'].apply(map_sfdm2)
    missing_count = df['sfdm2'].isnull().sum()
    print(f"缺失的'sfdm2'数量: {missing_count}")
    print(df['sfdm2'].value_counts())

    df.fillna(-1, inplace=True)
    print("缺失值已填-1")

    # 移除id列和目标列sfdm2
    X = df.drop(columns=['id', 'sfdm2'])
    y = df['sfdm2']

    # 标准化数据
    X = (X - X.mean()) / X.std()

    # 进行PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # 将PCA结果转换为DataFrame
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca['sfdm2'] = y

    df_pca.to_csv(f'{file_path}/support2_full_preprocessed_pca.csv', index=False)
    print(f"预处理后的数据（PCA）已保存为 support2_full_preprocessed_pca.csv")


@redirect_output(r"期末课程设计\output.txt")
@print_run_time
def main():

    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    file_path = "期末课程设计"

    download_data(file_path)

    # preprocess_data(file_path)

    # plot_stacked_bar_all_cols(file_path)

    pca = plot_pca_variance(file_path)

    # 选择主成分数量（例如，选择解释方差达到100%的主成分数量）
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 1) + 1
    print(f"选择的主成分数量: {n_components}")

    # 预处理数据并应用PCA
    preprocess_data_with_pca(file_path, n_components)

    # plot_stacked_bar_all_cols(file_path)

    # 定义分类器和参数网格
    classifiers = {
        'RandomForest': (RandomForestClassifier(random_state=42), {
            # 'max_depth': [None, 10, 20, 30],  # 尝试不同的最大深度
            # 'min_samples_split': [2, 5, 10],  # 尝试不同的最小拆分样本数
            # 'min_samples_leaf': [1, 2, 4],  # 尝试不同的最小叶样本数
            # 'n_estimators': [100, 200, 300, 400]  # 尝试不同的树数量
            'max_depth': [30],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'n_estimators': [400],
        }),
        'KNN': (KNeighborsClassifier(), {
            # 'n_neighbors': list(range(1, 31)),  # 尝试更多的邻居数量
            # 'weights': ['uniform', 'distance'],
            # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            # 'leaf_size': list(range(10, 51, 5)),  # 尝试不同的叶子大小
            # 'p': [1, 2]  # 尝试不同的距离度量（曼哈顿距离和欧几里得距离）
            'algorithm': ['auto'], 
            'leaf_size': [10],
            'n_neighbors': [1],
            'p': [1],
            'weights': ['uniform']
        }),
        'SVM-rbf': (SVC(random_state=42), {
            # 'C': [0.1, 1, 10, 100], # 尝试不同的正则化参数
            # 'kernel': ['linear', 'rbf', 'poly'], # 尝试不同的内核
            # 'gamma': ['scale'] # 尝试不同的gamma值
            # 'C': [10],
            # 'kernel': ['linear'],
            # 'gamma': ['scale']
            'C': [10],
            'kernel': ['rbf'],
            'gamma': ['scale']
        }),
        'SVM-linear': (LinearSVC(random_state=42, max_iter=10000), { # 专门优化线性速度
            # 'C': [0.1, 1, 10, 100]
            'C': [10]
        }),
        'DecisionTree': (DecisionTreeClassifier(random_state=42), {
            'max_depth': [None, 10, 20, 30], # 尝试不同的最大深度
            'min_samples_split': [2, 5, 10], # 尝试不同的最小拆
            'min_samples_leaf': [1, 2, 4] # 尝试不同的最小叶样本数
            # "max_depth": [30], 
            # "min_samples_leaf": [1], 
            # "min_samples_split": [2]
        }),
        'GradientBoosting': (GradientBoostingClassifier(random_state=42), {
            'n_estimators': [100, 200, 300], # 尝试不同的树数量
            'learning_rate': [0.01, 0.1, 0.2], # 尝试不同的学习率
            'max_depth': [3, 5, 7] # 尝试不同的最大深度
            # 'n_estimators': [300],
            # 'learning_rate': [0.2],
            # 'max_depth': [7] 
        }),
        'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {
            'C': [0.1, 1, 10, 100], # 尝试不同的正则化参数
            'penalty': ['l1', 'l2'], # 尝试不同的惩罚
            'solver': ['liblinear', 'saga'] # 尝试不同的求解器
            # 'C': [10],
            # 'penalty': ['l1'],
            # 'solver': ['liblinear']
        }),
        'GaussianNB': (GaussianNB(), {}),  # 高斯朴素贝叶斯不需要调参
        'BernoulliNB': (BernoulliNB(), {
            'alpha': [0.1, 0.5, 1.0, 2.0], # 平滑参数
            'binarize': [0.0, 0.5], # 二值化阈值
            'fit_prior': [True, False] # 是否学习先验概率
            # "alpha": [0.1], 
            # "binarize": [0.5], 
            # "fit_prior": [True]
        })
    }

    for name, (model, param_grid) in tqdm_notebook(classifiers.items(), desc="Classifiers"):
        print(f"Training and evaluating {name}...")
        y_val, y_val_pred, cm, cr, r2 = train_and_evaluate(file_path, 
                                                           model, 
                                                           param_grid, 
                                                           classifier_name=name, 
                                                           test_size=0.2, 
                                                           do_preproces=False)
        print(f"{name} 验证集分类报告：")
        print(cr)
        print(f"{name} 验证集R2得分：", r2)
        plot_confusion_matrix(cm, f'{name} 验证集混淆矩阵')

if __name__ == "__main__":
    main()