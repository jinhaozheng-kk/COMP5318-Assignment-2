# COMP5318 Assignment 2 — 图像分类：随机森林 vs 神经网络

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Classes-9-blue" alt="9 classes">
  <img src="https://img.shields.io/badge/Best%20CNN-77.68%25-brightgreen" alt="best accuracy">
</p>

COMP5318（Machine Learning）Assignment 2 **Group 131** 的完整提交：在课程提供的 9 类图像数据集上，系统比较**传统机器学习**（随机森林）与**神经网络**（全连接网络、卷积网络）在图像分类任务上的表现，并做超参数搜索与最终模型评估。

**最终测试准确率**：

| 模型 | 测试准确率 |
| --- | ---: |
| 随机森林（调参后） | 66.16% |
| 全连接网络（调参后） | 59.94% |
| **CNN（最佳配置）** | **77.68%** |

CNN 明显优于其他两种方法，说明对图像任务而言，卷积结构带来的空间特征提取能力比手工展平特征 + 树模型更有效。

**小组成员**：Group 131（SID1: 540580169、SID2: 530637868、SID3: 540881594）。本仓库为小组课程作业归档，非个人独立完成。

---

## 目录

- [1. 数据集与探索](#1-数据集与探索)
- [2. 预处理](#2-预处理)
- [3. 方法一：随机森林](#3-方法一随机森林)
- [4. 方法二：全连接神经网络](#4-方法二全连接神经网络)
- [5. 方法三：卷积神经网络](#5-方法三卷积神经网络)
- [6. 最终结果对比](#6-最终结果对比)
- [7. 逐类表现分析](#7-逐类表现分析)
- [8. 如何复现](#8-如何复现)
- [9. 仓库结构](#9-仓库结构)
- [10. 已知限制](#10-已知限制)
- [11. 参考](#11-参考)

---

## 1. 数据集与探索

课程提供的数据以 NumPy 数组形式存放（`data/` 目录）：

| 文件 | 形状 | 类型 | 说明 |
| --- | --- | --- | --- |
| `X_train.npy` | `(32000, 28, 28, 3)` | uint8 | 训练图像，RGB，取值 0–255 |
| `y_train.npy` | `(32000,)` | uint8 | 训练标签，0–8 |
| `X_test.npy` | `(8000, 28, 28, 3)` | uint8 | 测试图像 |
| `y_test.npy` | `(8000,)` | uint8 | 测试标签 |

- **9 个类别**（标签 0–8），训练/测试按 4:1 划分（32,000 / 8,000）；
- 类别分布**不均衡**：

| 类别 | 训练样本 | 测试样本 |
| ---: | ---: | ---: |
| 0 | 3,490 | 873 |
| 1 | 3,431 | 858 |
| 2 | 3,505 | 877 |
| 3 | 3,656 | 914 |
| 4 | 2,950 | 737 |
| 5 | 4,290 | 1,072 |
| 6 | 2,728 | 682 |
| 7 | 3,253 | 813 |
| 8 | 4,697 | 1,174 |
| **合计** | **32,000** | **8,000** |

类别 8 比类别 6 多出约 72%，这对准确率和 macro-F1 的差异有直接影响。

Notebook 里做了：形状与 dtype 检查、类别频次统计、每类随机样例可视化、像素值范围检查（0–255）。

## 2. 预处理

1. **归一化**：`X.astype('float32') / 255.0`，把像素压到 0–1；
2. **展平（仅树模型 / MLP）**：`reshape(N, -1)` → 每张图 28×28×3 = **2,352 维**特征；
3. **One-Hot 标签（神经网络）**：`OneHotEncoder` → `(N, 9)`；
4. **训练/验证划分（神经网络）**：`train_test_split(test_size=0.2, random_state=42, stratify=y_train)`，从训练集再切出 20% 做验证；
5. 代码里还有一步 `X_train_resized = resize(img, (28,28))`，但数据本来就是 28×28，**实际是空操作**，属于历史遗留代码。

> 注意：随机森林与 MLP 使用展平后的 2,352 维向量；CNN 使用原始 `(28, 28, 3)` 图像，不做展平。

## 3. 方法一：随机森林

```python
RandomForestClassifier(n_estimators=100, random_state=42)   # 基线
```

- 基线测试准确率：**0.6448**
- 基线分类报告（关键类）：类别 0/1 的 F1 高达 0.89/0.95，而类别 2/6 只有 0.40/0.19——说明这两类在像素空间上与其他类高度混淆。

**超参数搜索**（`GridSearchCV`，3 折，5 个参数共 **216 组配置 → 648 次拟合**）：

```python
param_grid = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}
```

- 最佳参数：`n_estimators=200, max_depth=30, max_features='sqrt', min_samples_split=5, min_samples_leaf=1`
- 最佳 CV 分数：**0.6469**
- 调参后测试准确率：**0.6616**（相比基线 +1.7 个百分点）
- 单次 fit 平均约 190–213 秒（CPU），整个网格搜索比较耗时

## 4. 方法二：全连接神经网络

### 4.1 基线 MLP

```python
create_mlp_model(hidden_layers=2, neurons=128, dropout_rate=0.2, learning_rate=0.001)
# Dense(128, relu) → Dropout(0.2) → Dense(128, relu) → Dropout(0.2) → Dense(9, softmax)
# Adam, categorical_crossentropy, batch=64, 10 epochs, validation_split=0.2
```

- 测试准确率：**0.3251**（明显欠拟合：10 轮不够，128 神经元也偏小）

### 4.2 MLP 超参数搜索

搜索空间（12 组配置）：

| 超参数 | 取值 |
| --- | --- |
| `hidden_layers` | 2 |
| `neurons` | 256, 512 |
| `dropout_rate` | 0.2, 0.3 |
| `learning_rate` | 5e-5, 1e-4, 1e-3 |

- 每组最多训练 **100 epochs**，`EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`；
- 整个搜索耗时约 **637 秒**；
- 最佳配置：`neurons=512, dropout=0.2, lr=5e-5`，最佳验证准确率 **0.6041**。

### 4.3 最终 MLP

用最佳参数在完整训练集上重训 **120 epochs**（早停、恢复最佳权重），测试准确率 **0.5994**。

分类报告显示类别 2/6/7 的 F1 只有 0.33/0.37/0.42，与随机森林的失败模式一致。

## 5. 方法三：卷积神经网络

### 5.1 基线 CNN

```python
Sequential([
    Conv2D(32, 3×3, relu), MaxPooling2D(2×2),
    Conv2D(64, 3×3, relu), MaxPooling2D(2×2),
    Flatten(), Dense(64, relu), Dropout(0.5), Dense(9, softmax),
])
# Adam lr=1e-3, batch=128, 10 epochs, validation_split=0.2
```

- 测试准确率：**0.7360**（比同轮数的 MLP 高出 40 多个百分点）

### 5.2 CNN 配置对比

固定上面的结构，比较 4 组配置（每组 20 epochs）：

| filters | dropout | learning rate | 测试准确率 | F1-macro | F1-weighted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.3 | 1e-3 | 0.7669 | 0.7500 | 0.7578 |
| 64 | 0.5 | 1e-3 | 0.7197 | 0.6986 | 0.7084 |
| 64 | 0.3 | 5e-4 | **0.7751** | **0.7699** | **0.7751** |
| 128 | 0.5 | 5e-4 | 0.7393 | 0.7091 | 0.7188 |

调参阶段分数最高的是 `(64, 0.3, 5e-4)`。

### 5.3 最终 CNN

最终模型使用 `filters=32, dropout=0.3, lr=1e-3` 的配置重新训练 20 epochs，测试准确率 **0.7768**（比调参阶段同配置的 0.7669 略高，属于正常波动）。

> 注：最终 CNN 选用的配置与调参阶段分数最高的 `(64, 0.3, 5e-4)` 不一致，这是原 notebook 的实际情况；重训后 32 filters 配置反而略高。若严格按"验证集最优"原则，应使用 `(64, 0.3, 5e-4)`，见第 10 节。

## 6. 最终结果对比

| 模型 | 关键配置 | 测试准确率 | 相对提升 |
| --- | --- | ---: | ---: |
| 随机森林（基线） | 100 树 | 0.6448 | — |
| 随机森林（调参后） | 200 树, depth 30 | 0.6616 | +1.68 pp |
| MLP（基线） | 2×128, 10 epochs | 0.3251 | — |
| MLP（调参后） | 2×512, dropout 0.2, lr 5e-5 | 0.5994 | +27.4 pp |
| CNN（基线） | 32/64, dropout 0.5, 10 epochs | 0.7360 | — |
| **CNN（最终）** | 32/64, dropout 0.3, lr 1e-3, 20 epochs | **0.7768** | **+4.08 pp** |

结论：

1. **CNN 最强**（77.68%），因为它保留了图像的空间结构，卷积核能学到局部纹理/边缘；
2. **随机森林**（66.16%）作为传统基线表现稳定，但对像素展平后的特征缺乏空间建模能力；
3. **MLP**（59.94%）最弱——全连接层参数量大、对图像平移不敏感，需要更多数据和正则化才能接近 CNN；
4. 三个模型都**难分类别一致**：类别 2、6、7 的 F1 明显偏低，说明这些类别在视觉上与其他类相似或样本较少。

## 7. 逐类表现分析

随机森林（调参后）与最终 MLP 的逐类 F1（测试集）：

| 类别 | RF（调参后）F1 | MLP（最终）F1 | 样本数 |
| ---: | ---: | ---: | ---: |
| 0 | 0.89 | 0.88 | 873 |
| 1 | 0.94 | 0.90 | 858 |
| 2 | 0.42 | 0.33 | 877 |
| 3 | 0.78 | 0.70 | 914 |
| 4 | 0.59 | 0.59 | 737 |
| 5 | 0.69 | 0.55 | 1,072 |
| 6 | 0.19 | 0.37 | 682 |
| 7 | 0.50 | 0.42 | 813 |
| 8 | 0.62 | 0.56 | 1,174 |

观察：

- 类别 0、1 在两种模型上都很好（F1 ≈ 0.9），说明它们视觉上区分度高；
- 类别 2 在 RF 上召回只有 0.31（precision 0.69），大量被漏判；
- 类别 6 是 RF 的灾难（F1 0.19，召回 0.12），MLP 稍好（0.37），两者都远低于其他类；
- 类别 5/8 样本多，召回尚可但精度一般，说明存在"把邻近类别判成它们"的误检；
- 不均衡类别（6 最少）表现最差，与训练样本不足一致。

## 8. 如何复现

### 环境

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：`numpy`、`pandas`、`matplotlib`、`seaborn`、`scikit-image`、`scikit-learn`、`tensorflow`、`jupyter`（**未锁定版本**，不同版本下结果可能有差异）。

### 运行

```bash
jupyter lab
# 打开 COMP5318_assignment2_Zheng_Tian_Ling.ipynb，从头执行
```

Notebook 直接读取 `data/` 下的 4 个 `.npy` 文件（仓库已包含，约 94 MB）。数据无需下载、无需联网。

### 耗时参考

- 随机森林网格搜索：216 组 × 3 折，CPU 上约 0.5–1 小时；
- MLP 超参搜索：12 组 × 最多 100 epochs，约 10 分钟（GPU 更快）；
- CNN 4 组配置 × 20 epochs + 最终重训：约 10 分钟（GPU）。

### 报告文件

`reports/` 里保留了提交用材料：

- `COMP5318_assignment2_Zheng_Tian_Ling.html`：notebook 的 HTML 导出（含所有输出与图表）；
- `COMP5318_assignment2_specification.pdf`：作业说明；
- `RF_CNN_report.docx`：书面报告。

## 9. 仓库结构

```text
COMP5318-Assignment-2/
├─ COMP5318_assignment2_Zheng_Tian_Ling.ipynb   # 主 notebook（数据探索 → 三模型 → 调参 → 最终模型）
├─ data/
│  ├─ X_train.npy   # 75 MB
│  ├─ X_test.npy    # 18 MB
│  ├─ y_train.npy
│  └─ y_test.npy
├─ reports/
│  ├─ COMP5318_assignment2_Zheng_Tian_Ling.html
│  ├─ COMP5318_assignment2_specification.pdf
│  └─ RF_CNN_report.docx
├─ requirements.txt
├─ .gitignore
└─ README.md
```

Notebook 共 53 个 cell，结构对应作业模板：Setup → 数据加载/探索/预处理 → 三种算法设计与调参 → 最终模型（独立重训）→ 报告。

## 10. 已知限制

- **数据集未提交来源说明**：`data/` 里的 `.npy` 是课程提供的数据，仓库没有记录数据集名称/来源与预处理方式；
- **最终 CNN 配置与调参最优不一致**：调参最优是 `(64, 0.3, 5e-4)`（0.7751），最终模型用了 `(32, 0.3, 1e-3)`（0.7768），严格来说不符合"按验证集选模型"的流程；
- **类别不均衡未处理**：没有 class weight、重采样或 focal loss，类别 6/2/7 的表现被拉低；
- **数据以原始数组直接提交**（94 MB，单文件最大 75 MB），已接近 GitHub 100 MB 限制，建议改用 Git LFS 或只保留下载脚本；
- `requirements.txt` 未锁定版本，TensorFlow / scikit-learn 版本差异会影响结果；
- Notebook 中保留了小组成员 SID，公开分享前建议匿名化；
- 预处理里的 `resize` 是空操作、`X_train_resized` 未被使用，属于历史遗留代码；
- 没有保存最终模型权重（`.h5`/`.keras` 已被 `.gitignore` 忽略），复现需重新训练；
- 仓库未附带 LICENSE 文件（课程作业，默认保留所有权利）。

## 11. 参考

- [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [TensorFlow / Keras Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
- [Dropout 论文](https://jmlr.org/papers/v15/srivastava14a.html)
- 课程：COMP4318/COMP5318 Assignment 2
