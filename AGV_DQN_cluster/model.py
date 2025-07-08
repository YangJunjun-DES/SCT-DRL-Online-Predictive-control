# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:23:54 2024

@author: 杨军军
"""
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split


load_data = sio.loadmat('.\\data\\AGV\\CaoBinData1.mat')  #Get the model file from Excel 
Env_data = load_data['CaoBinData1']
# 假设 A 是一个 numpy 数组，包含 800 行和 11 列
#也可以用以下代码来创建示例数据
# A = np.random.rand(800, 11)

# 将矩阵 A 的最后一列作为标签，其余部分作为特征
X = Env_data[:, :-1]
y = Env_data[:, -1]

# 使用 train_test_split 将数据拆分成训练集和测试集，保证每个标签种类的数据平均分布
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 将标签重新组合到特征矩阵中
train_data = np.column_stack((X_train, y_train))
test_data = np.column_stack((X_test, y_test))

# 获取训练集中的唯一标签
unique_labels = np.unique(y_train)

# 创建一个字典来存储按照标签拆分的训练数据
label_to_train_data = {label: [] for label in unique_labels}

# 按照标签拆分训练数据
for row in train_data:
    label = row[-1]
    label_to_train_data[label].append(row)

# 将列表转换为 numpy 数组
for label in label_to_train_data:
    label_to_train_data[label] = np.array(label_to_train_data[label])

# 输出每个标签对应的训练数据的形状
for label, data in label_to_train_data.items():
    print(f"Label {label}: Train data shape {data.shape}")

# 输出测试集的形状
print("Test data shape:", test_data.shape)


#The seprate matrix of label L0
i_matrix = label_to_train_data[0]



