import pandas as pd
import numpy as np

# 1. 读取 Excel 数据
file_path = r'C:\Users\EPR\Desktop\WH\index_change_new-plus-analysis.xlsx'
df = pd.read_excel(file_path)

# 假设 Excel 中第一行的列名为 'S', 'LCR', 'G'

# 2. 数据预处理：归一化
# 对于成本指标（S 和 LCR），归一化方法采用：
#    normalized_value = (max - x) / (max - min)
# 这样转换后，数值越大表示越好
# 对于效益指标（G），归一化方法采用：
#    normalized_value = (x - min) / (max - min)
# 保证各指标方向一致

def normalize_cost(series):
    return (series.max() - series) / (series.max() - series.min())

def normalize_benefit(series):
    return (series - series.min()) / (series.max() - series.min())

df_norm = pd.DataFrame()
df_norm['S'] = normalize_cost(df['S'])
df_norm['LCR'] = normalize_cost(df['LCR'])
df_norm['G'] = normalize_benefit(df['G'])

# 3. 计算各指标下各个样本的比重 p_ij
# 对于每个指标 j，p_ij = x_ij / (∑_i x_ij)
p = df_norm / df_norm.sum(axis=0)

# 4. 计算信息熵
# 定义常数 k = 1/ln(n)，其中 n 是样本数
n = df.shape[0]
k = 1 / np.log(n)

# 为避免 log(0) 的问题，当 p_ij 为 0 时认为其贡献为 0
entropy = {}
for col in p.columns:
    p_col = p[col].values
    # 对于 p==0 的情况，设定 log(p) 为 0（因为 0*log0 定义为 0）
    # 这里用 np.where 替换 0 为 1，log(1)=0
    p_col_nonzero = np.where(p_col == 0, 1, p_col)
    e = -k * np.sum(p_col * np.log(p_col_nonzero))
    entropy[col] = e

# 5. 计算差异系数 d_j = 1 - e_j
d = {col: 1 - entropy[col] for col in entropy}

# 6. 计算指标权重：w_j = d_j / (∑_j d_j)
sum_d = sum(d.values())
weights = {col: d[col] / sum_d for col in d}

# 输出计算结果
print("指标权重：")
for col, weight in weights.items():
    print(f"{col}: {weight:.4f}")
