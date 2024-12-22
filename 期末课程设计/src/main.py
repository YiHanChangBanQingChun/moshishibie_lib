'''
fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality
7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,5
'''
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm  # 新增
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from imblearn.over_sampling import SMOTE  # 用于处理类别不平衡
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def load_data(filepath):
    print("Loading data...")
    data = pd.read_csv(filepath)
    
    # 检查缺失值
    print("Missing values in original data:", data.isnull().sum().sum())
    
    # 保持 'quality' 为整数类型，不转换为字符串
    # 如果需要，可以使用 LabelEncoder 对标签进行编码
    # 但在当前情况下，保留原始整数标签
    # data['quality'] = data['quality'].astype(str)
    
    X = data.drop(['quality'], axis=1)
    y = data['quality']
    print("Data loaded successfully.")
    return X, y

def train_model(X_train, y_train):
    print("Starting model training...")
    param_grid = {
        'n_estimators': [500],
        'max_depth': [20],
        'min_samples_split': [2]
    }
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, n_jobs=-1, scoring='f1_macro', verbose=1)
    
    for _ in tqdm(range(1), desc="Grid Search Progress"):
        grid_search.fit(X_train, y_train)
    
    best_classifier = grid_search.best_estimator_
    print(f'Best Parameters: {grid_search.best_params_}')
    best_classifier.fit(X_train, y_train)
    print("Model training completed.")
    return best_classifier

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    print(f'Accuracy: {acc:.2f}')
    print(f'F1 Score (Macro): {f1:.2f}')
    print(f'Precision (Macro): {precision:.2f}')
    print(f'Recall (Macro): {recall:.2f}')
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 使用 model.classes_ 作为类别名称，确保与 shap_values 对齐
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    class_names = model.classes_
    plot_confusion_matrix(cm, class_names)
    
    # SHAP解释
    print("Generating SHAP values...")
    # 使用 TreeExplainer 并禁用 additivity 检查
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    
    # 调试信息：检查 model.classes_ 和 shap_values 的长度是否匹配
    print(f"Model classes: {model.classes_}")
    if isinstance(shap_values, list):
        print(f"Number of SHAP value sets: {len(shap_values)}")
    else:
        print("SHAP values are not a list, proceeding accordingly.")
    
    if isinstance(shap_values, list):
        for idx, cls in enumerate(model.classes_):
            if idx >= len(shap_values):
                print(f"Skipping SHAP plot for class {cls} as it is not present in shap_values.")
                continue
            # 检查 shap_values[idx] 的形状是否与 X_test 一致
            if hasattr(shap_values[idx], 'shape') and shap_values[idx].shape[1] != X_test.shape[1]:
                print(f"Shape mismatch for class {cls}: shap_values[{idx}].shape = {shap_values[idx].shape}, X_test.shape[1] = {X_test.shape[1]}")
                print(f"Skipping SHAP summary plot for class {cls} due to shape mismatch.")
                continue
            print(f"Generating SHAP summary plot for class {cls}...")
            plt.figure()
            shap.summary_plot(shap_values[idx], X_test, plot_type="bar", show=False)
            plt.title(f'SHAP Summary Plot for Class {cls}')
            plt.tight_layout()
            plt.show()
    else:
        # 如果 shap_values 不是列表，则直接生成汇总图
        print("Generating overall SHAP summary plot...")
        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        plt.show()

    # 验证数据完整性
    print("Checking for missing values after resampling and before SHAP...")
    print("Missing values in resampled data:", X_test.isnull().sum().sum())

    # SHAP Dependence Plots for top features
    # 计算每个特征在所有类别上的平均绝对 SHAP 值
    if isinstance(shap_values, list):
        feature_importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        feature_importances = np.mean(np.abs(shap_values), axis=0)  # 修改这里，移除 .values
    top_features_indices = np.argsort(feature_importances)[::-1][:5]
    top_features = X_test.columns[top_features_indices]
    
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_test, show=False)
        
        # 多类别 SHAP 值的平均
        if isinstance(shap_values, list):
            shap_vals_combined = np.mean([sv[:, X_test.columns.get_loc(feature)] for sv in shap_values], axis=0)  # 修改这里，移除 .values
        else:
            shap_vals_combined = shap_values[:, X_test.columns.get_loc(feature)]  # 修改这里，移除 .values
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_test[feature].values.reshape(-1, 1))
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, shap_vals_combined)
        shap_fit = lin_reg.predict(X_poly)
        
        sorted_idx = np.argsort(X_test[feature].values)
        plt.plot(X_test[feature].values[sorted_idx], shap_fit[sorted_idx], color='red', label=f'Fit: R²={lin_reg.score(X_poly, shap_vals_combined):.2f}')
        plt.legend()
        
        plt.title(f'SHAP Dependence Plot for {feature}')
        plt.tight_layout()
        plt.show()
    
    # 尝试不同的 SHAP 可视化方法
    # 使用 shap.force_plot 仅对少数样本进行
    print("Generating SHAP force plot for a single prediction...")
    shap.initjs()
    if isinstance(shap_values, list):
        shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True, show=False)
    else:
        shap.force_plot(explainer.expected_value, shap_values, X_test.iloc[0], matplotlib=True, show=False)
    plt.title('SHAP Force Plot for First Prediction')
    plt.tight_layout()
    plt.show()
    
    print("Model evaluation completed.")

def main():
    # 设置 Matplotlib 使用英文字体，避免字体问题
    plt.rcParams['font.sans-serif'] = ['Arial']  # 使用 Arial 或其他英文字体
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
    
    # 设置 Seaborn 主题
    sns.set(style='whitegrid')
    
    print("Program started...")
    local_path = r'D:\Users\admin\Documents\MATLAB\moshishibie_lib\期末课程设计\winequality-red.csv'
    X, y = load_data(local_path)
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 处理类别不平衡
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # 确保所有类别在重采样后都有样本
    print("Categories after SMOTE:", np.unique(y_resampled))
    
    # 检查重采样后的缺失值
    print("Checking for missing values after SMOTE...")
    print("Missing values in resampled data:", X_resampled.isnull().sum().sum())
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    print("Training set categories:", np.unique(y_train))
    print("Test set categories:", np.unique(y_test))
    
    best_classifier = train_model(X_train, y_train)
    evaluate_model(best_classifier, X_test, y_test)
    print("Program finished successfully.")

if __name__ == "__main__":
    main()