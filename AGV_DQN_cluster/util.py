# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:52:44 2022

@author: KaigeT
"""
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from sklearn.cluster import KMeans




# %% normalize AGV states
def AGV_norm_state(S):
    max_state = np.array([3, 7, 3, 5, 3, 1, 255])
    S_norm_arr = np.array(S)/max_state
    S_norm = S_norm_arr.tolist()
    S_norm_arr = np.array(S_norm)
    S_norm_arr = S_norm_arr[np.newaxis, :]
    return S_norm, S_norm_arr

# %% check AGV next states
def AGV_Next(State, action, param):
    # params
    AGV_1 = param.AGV_1
    AGV_2 = param.AGV_2
    AGV_3 = param.AGV_3
    AGV_4 = param.AGV_4
    AGV_5 = param.AGV_5
    SUP_IPSR = param.SUP_IPSR
    SUP_ZWSR = param.SUP_ZWSR
    
    X1 = State[0]   #from 0 
    X2 = State[1]
    X3 = State[2]
    X4 = State[3] 
    X5 = State[4]
    X6 = State[5]
    X7 = State[6]
    
    X1_ = np.where(AGV_1[X1, :, action] == 1)
    if len(X1_[0]) == 0:
        X1_ = X1
    else:
        X1_ = X1_[0][0]
       
    X2_ = np.where(AGV_2[X2, :, action] == 1)
    if len(X2_[0]) == 0:
        X2_ = X2
    else:
        X2_ = X2_[0][0]
       
       
    X3_ = np.where(AGV_3[X3, :, action] == 1)
    
    if len(X3_[0]) == 0:
       X3_ = X3
    else:
       X3_ = X3_[0][0]
        
        
    X4_ = np.where(AGV_4[X4, :, action] == 1)
    if len(X4_[0]) == 0:
        X4_ = X4
    else:
        X4_ = X4_[0][0]
       
       
    X5_ = np.where(AGV_5[X5, :, action] == 1)
    if len(X5_[0]) == 0:
        X5_ = X5
    else:
        X5_ = X5_[0][0]
        
          
    X6_ = np.where(SUP_IPSR[X6, :, action] == 1)   
    X6_ = X6_[0][0]
    
    X7_ = np.where(SUP_ZWSR[X7, :, action] == 1)
    X7_ = X7_[0][0]
    
    State_ = [X1_, X2_, X3_, X4_, X5_, X6_, X7_]
    
    return(State_)
    
# %% check available events
def AGV_Enb(state, DFA):
    Events = [];
    M = np.where(DFA == 1)
    N = M[0]   #current state
    O = np.where(N==state)
    Q = M[2]
    for i in O:
        Events.append(Q[i])
    
    return(Events)

# %% check permit states
def AGV_Permit(obs, param):
        
    AGV_1 = param.AGV_1
    AGV_2 = param.AGV_2
    AGV_3 = param.AGV_3
    AGV_4 = param.AGV_4
    AGV_5 = param.AGV_5
    SUP_IPSR = param.SUP_IPSR
    SUP_ZWSR = param.SUP_ZWSR
    
    Enable_P1 = AGV_Enb(obs[0], AGV_1)   #define Enb function
    Enable_P2 = AGV_Enb(obs[1], AGV_2)
    Enable_P3 = AGV_Enb(obs[2], AGV_3)
    Enable_P4 = AGV_Enb(obs[3], AGV_4)
    Enable_P5 = AGV_Enb(obs[4], AGV_5)
    
    Enable_P = np.union1d(Enable_P1, Enable_P2)
    Enable_P = np.union1d(Enable_P, Enable_P3)
    Enable_P = np.union1d(Enable_P, Enable_P4)
    Enable_P = np.union1d(Enable_P, Enable_P5)
        
    Enable_B1SUP = AGV_Enb(obs[5], SUP_IPSR)
    Enable_B2SUP = AGV_Enb(obs[6], SUP_ZWSR)    
    Enable = np.intersect1d(Enable_B1SUP, Enable_B2SUP)
    
    Enable_P_S = np.intersect1d(Enable_P, Enable)
    return(Enable_P_S,Enable_P)

#%% normalized variance
def normalized_variance(pattern_value):
    # 将输入转换为NumPy数组
    arr = np.array(pattern_value)
    if len(arr) == 0:
        normalized_u = 0  # 处理空数组情况
    else:    
        # 计算方差
        u = np.var(arr)
        
        # 获取数组的最大值和最小值
        max_val = np.max(arr)
        min_val = np.min(arr)
        
        # 处理所有元素相同的情况
        if max_val == min_val:
            normalized_u = 0
        else:        
            # 计算最大可能方差（数据分布在两极时的方差）
            max_var = ((max_val - min_val) ** 2) / 4
            
            # 归一化方差
            normalized_u = u / max_var
        
    return normalized_u

#%%
def normalize_and_std(pattern_value):
    # 归一化
    normalized_values = (pattern_value - np.min(pattern_value)) / (np.max(pattern_value) - np.min(pattern_value))
    
    # 计算标准差
    std_dev = np.std(normalized_values)
    
    return std_dev


# %%K-Mmeans
def get_larger_cluster_indices(pattern_value, K):
    # 将一维数据转换为二维格式
    X = np.array(pattern_value).reshape(-1, 1)
    # 使用K-means分2类（固定随机种子保证可重复性）
    kmeans = KMeans(n_clusters= K, n_init=10).fit(X)   
    #kmeans = KMeans(n_clusters= K, random_state=50, n_init=10).fit(X)
    # 计算3个簇的均值
    cluster_means = [X[kmeans.labels_ == i].mean() for i in range(K)]    
    # 确定较大值簇的标签
    larger_label = np.argmax(cluster_means)    
    # 返回较大簇的原始索引
    return [i for i, label in enumerate(kmeans.labels_) if label == larger_label]

# %% Cluster method 0
def best_selection(S, pattern_value, param, Dead_states):
    all_S_ = []
    all_Enb_ = []
    isDone_test = 0
    if S != [0, 0, 0, 0, 0, 0, 0]:
        [Enb, _,] = AGV_Permit(S, param)
        print(Enb)
        if set(Enb).isdisjoint(param.E_c):
            pattern = Enb
        else:
            pattern_index = np.argmax(pattern_value) #the action with maximum
            pattern = np.union1d(param.E_u, param.E_c[pattern_index])
            pattern = np.intersect1d(pattern, Enb)
    else:
        pattern= [0, 4]  #initial state
    print(pattern)    
    for event in pattern:
            S_ = AGV_Next(S, event, param)
            all_S_.append(S_)
            [Enb_, M] = AGV_Permit(S_, param)
            all_Enb_.append(Enb_)
    if any(len(vec) == 0 for vec in all_Enb_) or any(tuple(s) in set(map(tuple, all_S_)) for s in Dead_states):
            isDone_test = 1
    return all_S_, isDone_test, pattern
        
    
    
        

# %% Cluster method 1
def cluster_pattern(S, param, pattern_value, K, Delta):
    [Enb, Enable_P] = AGV_Permit(S, param)
    #print('Enabled events:', Enb, '\n') 
    all_S_ = []
    all_Enb_ = []
    isDone_test = 0       
    #Some special states lead to deadlocks by controllable events
    # if S == [0, 0, 2, 0, 3, 0, 113]: #(1956)   [0, 0, 2, 0, 2, 0, 65]
    #     pattern = [4,18]   #0 is disabled
    # elif S == [0, 0, 2, 0, 3, 0, 113]: #(2050)
    #     pattern = [4, 31] #0 is disabled
    # elif S == [0, 0, 2, 0, 0, 0, 65]:  #(240)
    #     pattern = [4]  # 0 is disabled
    # elif S == [0, 0, 0, 3, 2, 0, 205]:  #(3619)
    #     pattern = [0, 18] #4 disabled
    # elif S == [0, 0, 0, 3, 3, 0, 149]:  #(3713)
    #     pattern = [0, 31] #4 disabled
    # elif S == [0, 0, 0, 3, 0, 0, 205]:  #(1243)
    #     pattern = [0] #4 disabled
    # el
    
    if S == [0, 0, 0, 0, 0, 0, 0]:   
        pattern= [0, 4]  #initial state
    else:
        if pattern_value.all() < 0:
            pattern = np.intersect1d(param.E_u, Enb)
        elif set(Enb).isdisjoint(param.E_c) or len(set(pattern_value)) <= 1:
            pattern = Enb
        else:
            u = normalize_and_std(pattern_value) 
            if u <= Delta:  #All controllable are allowed
                    pattern = Enb
            else:
                pattern_index = get_larger_cluster_indices(pattern_value, K) #index of larger value
                selected_control_events = [param.E_c[i] for i in pattern_index]
                pattern = np.union1d(param.E_u, selected_control_events)
                pattern = np.intersect1d(pattern, Enb) 
    for event in pattern:
        S_ = AGV_Next(S, event, param)
        all_S_.append(S_)
        [Enb_, M] = AGV_Permit(S_, param)
        all_Enb_.append(Enb_)
    if any(len(vec) == 0 for vec in all_Enb_) or len(pattern) == 0:
        isDone_test = 1
    return all_S_, isDone_test
        
        
            



















