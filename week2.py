import pandas as pd
import numpy as np
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ----------------------
# 1. 数据预处理函数优化
# ----------------------
def preprocess_data(df):
    """
    1. 日期解析：指定格式，处理异常值
    2. 缺失值填充：针对所有分类列
    3. 高基数类别处理：合并稀有类别
    4. 特征衍生：年份、季节
    5. 冗余列删除
    """
    # 日期解析（指定格式，处理异常值）
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
    df = df.dropna(subset=['Date'])  # 删除日期解析失败的行
    
    # 衍生时间特征
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Season'] = df['Month'].apply(
        lambda x: 'Spring' if 3<=x<=5 else 
        'Summer' if 6<=x<=8 else 
        'Autumn' if 9<=x<=11 else 'Winter'
    )
    
    # 计算目标变量（平均价格）
    df['Average Price'] = (df['Low Price'] + df['High Price']) / 2
    
    # 缺失值填充（众数填充分类列）
    for col in ['Type', 'Item Size', 'Color']:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # 高基数类别处理（保留前10个常见类别，其余归为Other）
    for col in ['City Name', 'Origin']:
        top_10_cats = df[col].value_counts().head(10).index
        df[col] = df[col].apply(lambda x: x if x in top_10_cats else 'Other')
    
    # 删除冗余列
    drop_cols = [
        'Grade', 'Environment', 'Unit of Sale', 'Quality', 
        'Condition', 'Appearance', 'Storage', 'Crop', 
        'Trans Mode', 'Unnamed: 24', 'Unnamed: 25',
        'Low Price', 'High Price', 'Mostly Low', 'Mostly High',
        'Sub Variety', 'Origin District', 'Repack'
    ]
    df = df.drop(drop_cols, axis=1)
    
    return df

# ----------------------
# 5. 协方差分析（补充探索性分析）
# ----------------------
def analyze_covariance(df):
    """
    数值特征的协方差矩阵分析
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("⚠️ 无有效数值特征用于协方差分析")
        return
    
    df_numeric = df[numeric_cols].dropna()
    if df_numeric.empty:
        print("⚠️ 数值特征无有效数据（已删除所有缺失值行）")
        return
    
    cov_estimator = EmpiricalCovariance()
    cov_matrix = cov_estimator.fit(df_numeric).covariance_
    cov_df = pd.DataFrame(cov_matrix, columns=numeric_cols, index=numeric_cols)
    
    print("\n### 数值特征协方差分析 ###")
    print(cov_df)
    print(f"涉及特征：{numeric_cols}")

# ----------------------
# 主函数（整合数据分析阶段流程）
# ----------------------
def main():
    # 1. 加载并预处理数据
    df = pd.read_csv('US-pumpkins.csv')
    df_processed = preprocess_data(df)
    
    # 2. 协方差探索性分析
    analyze_covariance(df_processed)

if __name__ == "__main__":
    main()
