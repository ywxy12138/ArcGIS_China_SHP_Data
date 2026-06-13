# 城市人口流出预测项目 README（数据分析与处理全流程）

## 1. 项目目标
基于 2023 年城市经济与民生指标，预测 2024 年城市总流出人口数，并通过 SHAP 与 ALE 对模型进行可解释性分析。

- 预测目标：`2024年总流出人口数`
- 主要模型：XGBoost、RandomForest、LightGBM
- 解释方法：SHAP（特征贡献）、ALE（累积局部效应）

---

## 2. 项目目录核心文件

### 2.1 原始数据
- `2023_data.xlsx`
- `China_city_data_base_6.0.xlsx`
- `urban_migrant_outflow_2024.xls`

### 2.2 关键脚本
- `data_adjust.py`：数据融合、清洗、3sigma 处理
- `compare_models_rf_lgbm.py`：三模型统一口径对比评估
- `hyperparameter_optimization.py`：多策略超参优化（Grid/Random/Optuna）
- `optimize_models_hyperparams.py`：三模型定向参数优化与汇总
- `shap_analysis.py`：SHAP 分析（图表+报告）
- `ale_analysis_models.py`：ALE 计算（当前口径：仅测试集）
- `ale_visualization.py`：ALE 可视化汇总图生成

### 2.3 中间与结果文件（代表性）
- 合并与清洗：
  - `city_data_merged_2023_ml.xlsx`
  - `city_data_merged_2023_with_outflow_ml.csv`
  - `city_data_merged_2023_with_outflow_ml_clean_3sigma.csv`
- 训练/测试：
  - `city_data_train_stratified_8_2.csv`
  - `city_data_test_stratified_8_2.csv`
- 模型评估与优化：
  - `rf_lgbm_results_summary.csv`
  - `optimization_results.csv`
  - `optimization_summary.csv`
  - `optimized_hyperparameters.csv`
  - `FINAL_MODEL_COMPARISON.csv`
- SHAP：
  - `shap_feature_importance.csv`
  - `shap_importance_bar.png`
  - `shap_summary_scatter.png`
  - `shap_top4_dependence.png`
- ALE：
  - `ale_curve_details.csv`
  - `ale_feature_impact_summary.csv`
  - `ale_model_average_impact.csv`
  - `ale_model_comparison.png`
  - `ale_feature_model_heatmap.png`

---

## 3. 数据处理主链路（截至目前）

### 步骤 1：基础数据读取与字段规范化
脚本：`data_adjust.py`

主要逻辑：
1. 读取两张 2023 指标表和 2024 人口流出表。
2. 通过列名标准化函数处理中英文括号、空格、大小写等差异，提升列匹配鲁棒性。
3. 统一城市编码格式（去除 `.0` 等）用于连接。

产出：
- 第一阶段融合文件：`city_data_merged_2023_ml.xlsx`

### 步骤 2：目标变量拼接
脚本：`data_adjust.py`

主要逻辑：
1. 用 `来源城市ID` 与城市编码连接。
2. 将 `总流出人口数` 重命名为 `2024年总流出人口数`。

产出：
- `city_data_merged_2023_with_outflow_ml.xlsx`
- `city_data_merged_2023_with_outflow_ml.csv`

### 步骤 3：缺失值与异常值清洗（3sigma）
脚本：`data_adjust.py`

主要逻辑：
1. 指标列统一转数值，无法转换的值置为缺失。
2. 删除任一关键指标缺失的样本。
3. 对各指标按 $\mu \pm 3\sigma$ 区间筛除异常样本。

产出：
- `city_data_merged_2023_with_outflow_ml_clean_3sigma.xlsx`
- `city_data_merged_2023_with_outflow_ml_clean_3sigma.csv`

### 步骤 4：训练/测试集构建（已在当前仓库中存在）
当前项目后续脚本统一读取：
- `city_data_train_stratified_8_2.csv`
- `city_data_test_stratified_8_2.csv`

说明：
- 代码链路默认按 8:2 的分层抽样版本进行建模与解释。

---

## 4. 建模与优化流程

### 步骤 5：基线建模与统一口径评估
脚本：`compare_models_rf_lgbm.py`

主要逻辑：
1. 使用 21 个中文经济指标作为特征。
2. 采用 `SimpleImputer(median)` + 树模型 Pipeline。
3. 在测试集上计算 `R2` 与 `MAE`，并汇总输出。

产出：
- `rf_lgbm_results_summary.csv`

### 步骤 6：超参数优化（多策略）
脚本：`hyperparameter_optimization.py`

主要逻辑：
1. 提供 Grid Search / Random Search / Optuna（可选）三类策略。
2. 使用交叉验证指标（以 `R2` 为核心）选择参数组合。
3. 在测试集复核泛化表现。

产出（代表性）：
- `optimization_results.csv`
- `optimization_strategies_comparison.csv`
- `optimization_summary.csv`
- `FINAL_MODEL_COMPARISON.csv`

### 步骤 7：定向参数优化（三模型）
脚本：`optimize_models_hyperparams.py`

主要逻辑：
1. 针对三模型分别设计参数组合搜索。
2. 同时记录 CV 与测试集表现。
3. 输出各模型最佳参数与综合对比。

产出：
- `optimized_hyperparameters.csv`
- `four_scheme_results_summary.csv`
- `OPTIMIZATION_REPORT.txt`

---

## 5. 可解释性分析流程

### 步骤 8：SHAP 解释（全局特征贡献）
脚本：`shap_analysis.py`（以及 `shap_analysis_detailed_chinese.py`）

主要逻辑：
1. 使用训练集训练 XGBoost 管道。
2. 使用测试集计算 SHAP 值矩阵。
3. 生成特征重要性条形图、summary 散点图、dependence 图。

产出：
- `shap_feature_importance.csv`
- `shap_importance_bar.png`
- `shap_summary_scatter.png`
- `shap_top4_dependence.png`
- `SHAP_ANALYSIS_REPORT.txt`

### 步骤 9：ALE 计算（当前已调整为“仅测试集”）
脚本：`ale_analysis_models.py`

当前口径（非常重要）：
1. 模型训练仅使用训练集。
2. ALE 计算仅使用测试集特征分布（不使用测试标签参与训练）。

1D-ALE 计算逻辑：
1. 对单个特征按分位数分箱（默认 10 箱）。
2. 对每个区间内样本，将该特征替换为区间上下界，计算预测差值得到局部效应。
3. 对局部效应做累积，得到累积局部效应。
4. 对 ALE 曲线做中心化（样本均值为 0），便于跨模型比较。

产出：
- `ale_curve_details.csv`（分箱边界、局部效应、累积效应、中心化累积效应）
- `ale_feature_impact_summary.csv`（各模型各特征平均影响统计）
- `ale_model_average_impact.csv`（模型层面的平均影响对比）
- `ale_top6_xgboost.png`
- `ale_top6_randomforest.png`
- `ale_top6_lightgbm.png`

### 步骤 10：ALE 可视化汇总
脚本：`ale_visualization.py`

主要逻辑：
1. 读取 ALE 三张结果表。
2. 生成模型对比、热力图、曲线对比、统计散点、箱线图。
3. 生成文本汇总。

产出：
- `ale_model_comparison.png`
- `ale_feature_ranking_by_model.png`
- `ale_feature_model_heatmap.png`
- `ale_curve_feature_comparison.png`
- `ale_statistics_scatter.png`
- `ale_model_distribution_boxplot.png`
- `ALE_VISUALIZATION_SUMMARY.txt`

---

## 6. 推荐执行顺序（复现实验）

在项目根目录按以下顺序执行：

```bash
python data_adjust.py
python compare_models_rf_lgbm.py
python hyperparameter_optimization.py
python optimize_models_hyperparams.py
python shap_analysis.py
python ale_analysis_models.py
python ale_visualization.py
```

说明：
- 若仅复现可解释性部分，可从已有训练/测试集直接执行最后三步。
- 当前环境建议使用已配置的 `mytest` conda 环境。

---

## 7. 特征与目标口径

- 特征数：21（经济、财政、教育、医疗、交通、开放度等）
- 目标：`2024年总流出人口数`
- 评估指标：`R2`、`MAE`
- 可解释性指标：`Mean|SHAP|`、`mean_abs_ale`、`ale_range`

---

## 8. 当前版本关键约定

1. ALE 解释口径已固定为“仅测试集分布”。
2. SHAP 解释口径为“训练模型 + 测试集解释”。
3. 所有关键结果文件均为 UTF-8（或 UTF-8-SIG）编码，便于中文读取。
4. 图像中文字体在个别环境可能有告警（不影响数值结果）。

---

## 9. 常见问题与排查

1. 找不到训练/测试集文件：
   - 先确认 `city_data_train_stratified_8_2.csv` 与 `city_data_test_stratified_8_2.csv` 已存在。

2. 读取 `.xls` 失败：
   - 安装 `xlrd` 后重试。

3. SHAP 库缺失：
   - 安装 `shap` 后重试。

4. 中文图形显示方块/乱码：
   - 为 matplotlib 配置可用中文字体（不影响计算结果）。

---

## 10. 参考脚本与说明文件

- 处理逻辑：`data_adjust.py`
- 模型与优化：`compare_models_rf_lgbm.py`、`hyperparameter_optimization.py`、`optimize_models_hyperparams.py`
- 解释分析：`shap_analysis.py`、`ale_analysis_models.py`、`ale_visualization.py`
- 辅助说明：`FILE_INDEX_AND_USAGE_GUIDE.txt`、`SHAP_CODE_NAVIGATION_GUIDE.txt`、`实验过程.txt`
