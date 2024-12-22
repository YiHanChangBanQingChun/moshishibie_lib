import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# 加载数据
file_path = r'D:\Users\admin\Documents\MATLAB\moshishibie_lib\期末课程设计\support2_full_preprocessed.csv'
df = pd.read_csv(file_path)

# 定义特征和目标变量
X = df.drop(columns=['id', 'sfdm2'])
y = df['sfdm2']

# 分割数据集，80%训练，20%验证
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 训练随机森林分类器
model = RandomForestClassifier(
    random_state=42,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=1,
    n_estimators=400,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 使用SHAP进行解释
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_val)

# 绘制SHAP值的各种图

# 1. Waterfall图
shap.plots.waterfall(shap_values[0])

# 2. Force plot
shap.initjs()
shap.force_plot(shap_values[0].base_values, shap_values[0].values, X_val.iloc[0,:])

# 3. 特征影响图
shap.summary_plot(shap_values, X_val)

# 4. 特征依赖图（dependence scatter plot）
shap.dependence_plot("age", shap_values.values, X_val)

# 5. 特征密度散点图（beeswarm）
shap.plots.beeswarm(shap_values)

# 6. 特征重要性SHAP值
shap.summary_plot(shap_values, X_val, plot_type="bar")

# 7. 样本聚类下特征分布热力图
shap.plots.heatmap(shap_values)

# 8. 特征的层次聚类
shap.plots.bar(shap_values)

# 9. 多样本-不同特征SHAP决策图
shap.plots.scatter(shap_values[:, "age"], color=shap_values)

# 显示所有图
plt.show()