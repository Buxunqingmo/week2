# 美国南瓜价格预测项目

这是一个基于Python构建的美国南瓜价格预测项目，利用机器学习模型对美国南瓜市场价格进行预测和分析。

## 项目目录结构

```
US-Pumpkins-Price-Prediction/
├── US-pumpkins.csv # 原始数据集
├── week2.py # 主程序文件
├── model_comparison.png # 模型比较图表
├── feature_importance.png # 特征重要性图表
├── residuals/ # 各模型残差图
├── learning_curves/ # 各模型学习曲线
└── README.md # 项目说明文件
```

## 项目背景与目的

美国南瓜市场价格受多种因素影响，包括季节、品种、产地、市场需求等。本项目旨在：

1. 分析影响南瓜价格的关键因素
2. 构建高效的价格预测模型
3. 为南瓜种植者和经销商提供价格参考
4. 探索不同机器学习模型在价格预测任务中的表现

## 数据集说明

### 原始数据集
原始数据集包含1757条记录，26个特征。

#### 数据特征

| 特征名称 | 类型 | 非空值 | 缺失值 | 说明 |
|---------|------|------|------|------|
| City Name | object | 1757 | 0 | 待补充 |
| Type | object | 45 | 1712 | 待补充 |
| Package | object | 1757 | 0 | 待补充 |
| Variety | object | 1752 | 5 | 待补充 |
| Sub Variety | object | 296 | 1461 | 待补充 |
| Grade | float64 | 0 | 1757 | 待补充 |
| Date | object | 1757 | 0 | 待补充 |
| Low Price | float64 | 1757 | 0 | 待补充 |
| High Price | float64 | 1757 | 0 | 待补充 |
| Mostly Low | float64 | 1654 | 103 | 待补充 |
| Mostly High | float64 | 1654 | 103 | 待补充 |
| Origin | object | 1754 | 3 | 待补充 |
| Origin District | object | 131 | 1626 | 待补充 |
| Item Size | object | 1478 | 279 | 待补充 |
| Color | object | 1141 | 616 | 待补充 |
| Environment | float64 | 0 | 1757 | 待补充 |
| Unit of Sale | object | 162 | 1595 | 待补充 |
| Quality | float64 | 0 | 1757 | 待补充 |
| Condition | float64 | 0 | 1757 | 待补充 |
| Appearance | float64 | 0 | 1757 | 待补充 |
| Storage | float64 | 0 | 1757 | 待补充 |
| Crop | float64 | 0 | 1757 | 待补充 |
| Repack | object | 1757 | 0 | 待补充 |
| Trans Mode | float64 | 0 | 1757 | 待补充 |
| Unnamed: 24 | float64 | 0 | 1757 | 待补充 |
| Unnamed: 25 | object | 103 | 1654 | 待补充 |

#### 数据预览

原始数据集前5条记录：

```
City Name|Type|Package|Variety|Sub Variety|Grade|Date|Low Price|High Price|Mostly Low|Mostly High|Origin|Origin District|Item Size|Color|Environment|Unit of Sale|Quality|Condition|Appearance|Storage|Crop|Repack|Trans Mode|Unnamed: 24|Unnamed: 25|
|BALTIMORE|nan|24 inch bins|nan|nan|nan|4/29/17|270.0|280.0|270.0|280.0|MARYLAND|nan|lge|nan|nan|nan|nan|nan|nan|nan|nan|E|nan|nan|nan|
|BALTIMORE|nan|24 inch bins|nan|nan|nan|5/6/17|270.0|280.0|270.0|280.0|MARYLAND|nan|lge|nan|nan|nan|nan|nan|nan|nan|nan|E|nan|nan|nan|
|BALTIMORE|nan|24 inch bins|HOWDEN TYPE|nan|nan|9/24/16|160.0|160.0|160.0|160.0|DELAWARE|nan|med|ORANGE|nan|nan|nan|nan|nan|nan|nan|N|nan|nan|nan|
|BALTIMORE|nan|24 inch bins|HOWDEN TYPE|nan|nan|9/24/16|160.0|160.0|160.0|160.0|VIRGINIA|nan|med|ORANGE|nan|nan|nan|nan|nan|nan|nan|N|nan|nan|nan|
|BALTIMORE|nan|24 inch bins|HOWDEN TYPE|nan|nan|11/5/16|90.0|100.0|90.0|100.0|MARYLAND|nan|lge|ORANGE|nan|nan|nan|nan|nan|nan|nan|N|nan|nan|nan|
|```

### 预处理后数据集
预处理后数据集包含1757条记录，12个特征。

#### 数据预处理步骤
1. 日期解析与异常值处理
2. 衍生时间特征（月份、年份、季节）
3. 缺失值填充（使用众数填充分类特征）
4. 高基数类别处理（合并稀有类别）
5. 冗余特征删除

#### 预处理后数据预览

预处理后数据集前5条记录：

```
City Name|Type|Package|Variety|Date|Origin|Item Size|Color|Month|Year|Season|Average Price|
|BALTIMORE|Organic|24 inch bins|nan|2017-04-29|MARYLAND|lge|ORANGE|4|2017|Spring|275.0|
|BALTIMORE|Organic|24 inch bins|nan|2017-05-06|MARYLAND|lge|ORANGE|5|2017|Spring|275.0|
|BALTIMORE|Organic|24 inch bins|HOWDEN TYPE|2016-09-24|Other|med|ORANGE|9|2016|Autumn|160.0|
|BALTIMORE|Organic|24 inch bins|HOWDEN TYPE|2016-09-24|Other|med|ORANGE|9|2016|Autumn|160.0|
|BALTIMORE|Organic|24 inch bins|HOWDEN TYPE|2016-11-05|MARYLAND|lge|ORANGE|11|2016|Autumn|95.0|
|```

## 技术栈

本项目使用的主要技术和库如下：

- **数据处理**：Pandas, NumPy
- **数据可视化**：Matplotlib, Seaborn
- **机器学习**：Scikit-learn, XGBoost
  - 数据预处理：StandardScaler, OneHotEncoder, ColumnTransformer
  - 特征工程：特征选择、特征衍生
  - 模型：线性回归、岭回归、Lasso回归、K近邻、支持向量机、随机森林、XGBoost
  - 评估：均方误差(MSE)、平均绝对误差(MAE)、决定系数(R²)
- **开发环境**：Python 3.12.7

## 方法与步骤

### 1. 数据预处理
- 日期解析：将字符串日期转换为datetime格式，并处理解析错误
- 特征衍生：从日期中提取月份、年份，并根据月份确定季节
- 目标变量创建：根据最低价格和最高价格计算平均价格
- 缺失值处理：使用众数填充分类特征的缺失值
- 高基数类别处理：将出现频率低的类别合并为'Other'
- 特征选择：删除冗余和无关特征

### 2. 特征工程
- 数值特征：标准化处理（Month, Year）
- 分类特征：独热编码处理（City Name, Type, Package, Variety, Origin, Item Size, Color, Season）
- 特征处理管道：使用ColumnTransformer构建统一的特征处理流程

### 3. 模型训练与评估
- 数据集划分：按8:2比例划分训练集和测试集
- 模型选择：使用7种不同类型的回归模型
- 超参数调优：对每个模型使用GridSearchCV进行超参数优化
- 模型评估：使用MSE、MAE和R²作为评估指标
- 可视化分析：绘制模型比较图、特征重要性图、残差图和学习曲线

## 模型与评估指标

本项目评估了以下模型：

- 线性回归：基本线性模型
- 岭回归：带L2正则化的线性模型
- Lasso回归：带L1正则化的线性模型
- K近邻：基于实例的学习方法
- 支持向量机：使用不同核函数的非线性模型
- 随机森林：集成学习方法，基于决策树
- XGBoost：高效的梯度提升树算法

### 评估指标说明
- **均方误差(MSE)**：预测值与真实值差值平方的平均值，值越小表示预测越准确
- **平均绝对误差(MAE)**：预测值与真实值绝对差值的平均值，直观反映预测误差大小
- **决定系数(R²)**：衡量模型对数据的拟合程度，值范围[0,1]，值越大表示拟合越好

## 结果分析

### 1. 模型性能比较
通过对比各模型在测试集上的表现，可以发现：

1. 集成学习模型（随机森林、XGBoost）通常表现优于单一模型
2. 非线性模型（支持向量机、K近邻）在某些情况下可能优于线性模型
3. 正则化模型（岭回归、Lasso回归）相比普通线性回归有更好的泛化能力

### 2. 特征重要性分析
通过随机森林模型的特征重要性分析，发现影响南瓜价格的关键因素包括：

1. 季节(Season)：不同季节南瓜产量和需求的变化显著影响价格
2. 品种(Variety)：不同品种的南瓜具有不同的市场价值
3. 产地(Origin)：南瓜的产地影响运输成本和市场供应
4. 包装(Package)：不同包装方式可能反映产品的市场定位
5. 月份(Month)：价格随月份变化呈现季节性波动

### 3. 模型误差分析
通过残差图分析，可以观察到：

1. 部分模型在价格较高区域预测误差较大
2. 残差分布是否符合正态分布，反映模型假设的合理性
3. 是否存在系统性偏差，如预测值普遍高于或低于真实值

### 4. 学习曲线分析
学习曲线可以帮助我们：

1. 识别模型是否存在高偏差（欠拟合）或高方差（过拟合）
2. 确定是否需要更多训练数据来提高模型性能
3. 评估当前模型复杂度是否合适

## 运行指南

### 环境要求

- Python 3.8+
- 所需依赖包：
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost

### 安装依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 运行代码

1. 将数据集`US-pumpkins.csv`与代码`week2.py`放在同一目录下
2. 打开终端，进入代码所在目录
3. 运行以下命令：

```bash
python week2.py
```

4. 运行结果将生成：
   - 模型比较图表（model_comparison.png）
   - 特征重要性图表（feature_importance.png）
   - 各模型残差图（位于residuals目录）
   - 各模型学习曲线（位于learning_curves目录）
   - 控制台输出各模型评估指标

## 项目改进方向

1. **特征工程优化**：
   - 探索更多衍生特征，如价格趋势、市场供需关系等
   - 尝试特征交互作用，捕捉特征之间的复杂关系

2. **模型优化**：
   - 尝试更高级的集成学习方法，如Stacking
   - 考虑时间序列特性，使用时间序列模型

3. **数据增强**：
   - 收集更多相关数据，如天气数据、经济指标等
   - 考虑使用数据生成技术扩充数据集

4. **部署与应用**：
   - 构建Web应用程序，提供价格预测服务
   - 实现实时数据更新和预测功能

## 生成信息

本README文件由Python脚本自动生成于 2025年07月01日 13:32:51