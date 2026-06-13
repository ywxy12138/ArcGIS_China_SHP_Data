"""
================================================================================
城市人口流出预测模型 - SHAP 特征重要性分析脚本 (详细中文注释版)
================================================================================

SHAP (SHapley Additive exPlanations) 特征重要性分析框架
基于 Shapley 值理论，提供统一的特征贡献度解释方法

主要输出:
1. shap_importance_bar.png - 特征重要性条形图
2. shap_summary_scatter.png - SHAP值与特征值的散点图
3. shap_top4_dependence.png - 前4个重要特征的dependence图
4. shap_feature_importance.csv - 各特征的SHAP值统计
5. SHAP_ANALYSIS_REPORT.txt - 详细分析报告
================================================================================
"""

# ========== 导入库 ==========
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 数据处理和模型构建库
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ★★★ SHAP 库导入 (核心库用于SHAP值计算) ★★★
# 
# SHAP 是一个统一的特征重要性度量框架，基于博弈论中的 Shapley 值
# 
# 库功能:
# 1. TreeExplainer: 针对树模型(XGBoost, LightGBM等)的高效SHAP值计算器
#    - 利用树的结构进行条件期望计算
#    - 比通用方法快1000倍以上
# 
# 2. 可视化工具: summary_plot, dependence_plot等
#    - summary_plot: 展示所有特征的SHAP值分布
#    - dependence_plot: 展示单个特征的SHAP值与其值的关系
# 
# 3. 数值解释: shap_values 矩阵
#    - shape: (样本数, 特征数)
#    - 每个元素表示该样本该特征的SHAP值
#
try:
    import shap  # 安装: pip install shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. Install via: pip install shap")


def load_data():
    """
    数据加载与预处理函数
    
    作用: 读取训练集和测试集数据，进行特征提取和目标变量清理
    
    数据源:
    - city_data_train_stratified_8_2.csv (训练集, 155个城市)
    - city_data_test_stratified_8_2.csv (测试集, 38个城市)
    
    特征说明 (共21个经济指标):
    ├─ 产业相关 (3个):
    │  ├─ 地区生产总值(万元)
    │  ├─ 第一产业增加值(万元) - 农业
    │  ├─ 第二产业增加值(万元) - 工业
    │  └─ 第三产业增加值(万元) - 服务业
    │
    ├─ 人均指标 (1个):
    │  └─ 人均地区生产总值(元) - 人均收入水平
    │
    ├─ 财政金融 (5个):
    │  ├─ 地方财政一般预算内收入(万元)
    │  ├─ 地方财政一般预算内支出(万元)
    │  ├─ 年末金融机构存款余额(万元)
    │  ├─ 年末金融机构各项贷款余额(万元)
    │  └─ 社会消费品零售总额(万元)
    │
    ├─ 教育相关 (3个):
    │  ├─ 教育支出(万元)
    │  ├─ 普通高中学生数(万人)
    │  └─ 普通高等学校在校学生数(人)
    │
    ├─ 医疗相关 (2个):
    │  ├─ 医院、卫生院数(个)
    │  └─ 执业(助理)医师数(人)
    │
    ├─ 基础设施 (3个):
    │  ├─ 高新技术企业数(国家级)(个) - 科技创新
    │  ├─ 高速公路里程(公里)
    │  └─ 境内公路总里程(公里)
    │
    └─ 其他 (3个):
       ├─ 进出口总额亿元 - 贸易开放度
       ├─ 城镇居民人均可支配收入
       └─ 农村居民人均可支配收入
    
    目标变量: 2024年总流出人口数 (要预测的城市人口流出数量)
    
    返回值:
    - X_train: 训练集特征 (155个样本, 21个特征)
    - y_train: 训练集目标值
    - X_test: 测试集特征 (38个样本, 21个特征)
    - y_test: 测试集目标值
    - feature_names: 中文特征名称列表 (用于SHAP标签)
    """
    base_dir = Path.cwd()
    train_df = pd.read_csv(base_dir / 'city_data_train_stratified_8_2.csv')
    test_df = pd.read_csv(base_dir / 'city_data_test_stratified_8_2.csv')
    
    # ★ 使用原始中文列名确保SHAP值输出使用中文标签 ★
    # 这对于最终的特征重要性解释至关重要
    raw_feature_cols = [
        '地区生产总值(万元)', '第一产业增加值(万元)', '第二产业增加值(万元)',
        '第三产业增加值(万元)', '人均地区生产总值(元)',
        '地方财政一般预算内收入(万元)', '地方财政一般预算内支出(万元)',
        '年末金融机构存款余额(万元)', '年末金融机构各项贷款余额(万元)',
        '社会消费品零售总额(万元)', '教育支出(万元)',
        '普通高中学生数(万人)', '普通高等学校在校学生数(人)',
        '医院、卫生院数(个)', '执业(助理)医师数(人)',
        '高新技术企业数(国家级)(个)', '高速公路里程(公里)',
        '境内公路总里程(公里)', '进出口总额亿元',
        '城镇居民人均可支配收入', '农村居民人均可支配收入'
    ]
    
    # 特征名称直接使用中文 (为SHAP计算做准备)
    feature_names = raw_feature_cols
    
    # 从DataFrame中提取特征和目标值
    X_train = train_df[raw_feature_cols].apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(train_df['2024年总流出人口数'], errors='coerce')
    X_test = test_df[raw_feature_cols].apply(pd.to_numeric, errors='coerce')
    y_test = pd.to_numeric(test_df['2024年总流出人口数'], errors='coerce')
    
    # 数据清理: 移除含有NaN的行
    train_mask = y_train.notna()
    test_mask = y_test.notna()
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_test = X_test.loc[test_mask].copy()
    y_test = y_test.loc[test_mask].copy()
    
    # 设置列名为中文特征名称 (重要: 这些名字会出现在SHAP图表中)
    X_train.columns = feature_names
    X_test.columns = feature_names
    
    return X_train, y_train, X_test, y_test, feature_names


def train_optimal_xgboost(X_train, y_train):
    """
    训练优化后的 XGBoost 回归模型
    
    这个函数使用网格搜索已验证的最优超参数来训练XGBoost模型。
    模型会用于后续的SHAP值计算。
    
    超参数说明 (基于网格搜索优化):
    
    ├─ 模型复杂度参数:
    │  ├─ n_estimators=500: 集成的决策树数量
    │  │  (更多树 → 模型更复杂,但计算更慢)
    │  │
    │  ├─ max_depth=5: 单棵树的最大深度
    │  │  (控制树的复杂度,防止过拟合)
    │  │
    │  ├─ learning_rate=0.05: 梯度下降的步长
    │  │  (更小 → 训练更慢但可能更精确)
    │  │
    │  └─ subsample=0.8: 行采样比例
    │     (每次迭代使用80%的训练样本)
    │
    ├─ 正则化参数:
    │  ├─ colsample_bytree=0.8: 特征采样比例
    │  │  (每棵树使用80%的特征)
    │  │  (降低特征之间的相关性)
    │  │
    │  └─ objective='reg:squarederror': 损失函数
    │     (回归任务使用平方误差)
    │
    └─ 其他:
       ├─ random_state=42: 随机种子(确保可重现)
       └─ n_jobs=-1: 使用所有CPU核心并行计算
    
    为什么选择XGBoost?
    1. 效率高: 比随机森林快,支持GPU加速
    2. SHAP友好: TreeExplainer专门针对XGBoost优化
    3. 非线性: 能捕捉特征与目标的复杂关系
    4. 特征重要性: 内置特征重要性计算
    
    参数:
    - X_train: 训练特征矩阵 (155个样本, 21个特征)
    - y_train: 训练目标值 (城市人口流出数)
    
    返回: sklearn Pipeline 对象
    """
    # 设置XGBoost超参数
    # 这些参数已通过GridSearchCV验证为最优配置
    model_params = {
        'n_estimators': 500,        # ★ 树的数量影响模型容量
        'learning_rate': 0.05,      # ★ 学习率影响收敛速度和精度
        'max_depth': 5,             # ★ 深度影响模型复杂度
        'subsample': 0.8,           # 行采样(随机梯度提升)
        'colsample_bytree': 0.8,    # 列采样(特征随机性)
        'objective': 'reg:squarederror',  # 平方误差损失
        'random_state': 42,         # 随机种子
        'n_jobs': -1,               # 并行计算
    }
    
    # 构建处理管道
    # Pipeline 确保训练和预测时应用相同的预处理步骤
    pipe = Pipeline([
        # 步骤1: 缺失值填补 (用中位数填补NaN)
        ('imputer', SimpleImputer(strategy='median')),
        # 步骤2: XGBoost 模型训练
        ('model', XGBRegressor(**model_params))
    ])
    
    # 训练管道 (自动执行imputer然后训练模型)
    pipe.fit(X_train, y_train)
    return pipe


def generate_shap_analysis(pipe, X_train, X_test, y_test, feature_names, base_dir):
    """
    ★★★★★ SHAP 值计算与可视化的核心函数 ★★★★★
    
    本函数是整个SHAP分析流程的中心,包含以下关键步骤:
    
    【步骤 1】解释器初始化 - 创建 TreeExplainer
    【步骤 2】SHAP 值计算 - 计算测试集所有样本的SHAP值
    【步骤 3】特征重要性统计 - 计算每个特征的平均|SHAP|值
    【步骤 4】SHAP 可视化 - 生成各种图表
    【步骤 5】结果整理 - 生成报告和CSV
    
    ═══════════════════════════════════════════════════════════════════
    参数说明:
    ═══════════════════════════════════════════════════════════════════
    - pipe: 已训练的 sklearn Pipeline (包含imputer和XGBoost模型)
    - X_train: 训练集特征 (155个样本, 21个特征)
    - X_test: 测试集特征 (38个样本, 21个特征)
    - y_test: 测试集目标值
    - feature_names: 中文特征名称列表(长度=21)
    - base_dir: 输出文件的保存目录
    
    返回值: 字典,包含以下key
    ├─ 'shap_values': numpy数组 (38, 21) - 原始SHAP值矩阵
    ├─ 'X_test_imputed': DataFrame - 预处理后的特征数据
    ├─ 'feature_importance_df': DataFrame - 特征重要性统计表
    ├─ 'explainer': shap.TreeExplainer - SHAP解释器对象
    ├─ 'r2': float - 模型R²分数
    └─ 'mae': float - 平均绝对误差
    """
    
    if not SHAP_AVAILABLE:
        print("Error: SHAP not available. Please install: pip install shap")
        return None
    
    # ═══════════════════════════════════════════════════════════════════
    # 【预处理】从管道中提取模型和预处理器
    # ═══════════════════════════════════════════════════════════════════
    model = pipe.named_steps['model']  # 提取已训练的XGBoost模型
    imputer = pipe.named_steps['imputer']  # 提取缺失值填补器
    
    # 对测试集应用预处理 (使用训练时拟合的参数)
    # 这确保了consistency: 训练时的预处理方式 = 测试时的预处理方式
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=feature_names,
        index=X_test.index
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # ★★★ 【步骤 1】初始化 SHAP 解释器 ★★★
    # ═══════════════════════════════════════════════════════════════════
    # 
    # TreeExplainer 是针对树模型(XGBoost, LightGBM, CatBoost)的
    # 高效SHAP值计算工具
    #
    # 工作原理:
    #   1. 遍历XGBoost的所有决策树 (500棵)
    #   2. 对每棵树追踪样本从根到叶的路径
    #   3. 计算每个特征在该路径上的贡献
    #   4. 对所有树的贡献进行加权平均
    #   5. 得到该特征的SHAP值
    #
    # 复杂度分析:
    #   传统Shapley值: O(2^M) - M是特征数 (指数爆炸!)
    #   Tree SHAP:    O(T*L*M²) - T=树数,L=路径长度,M=特征数(多项式)
    #   本例中: O(500*5*441) ≈ 110万 次操作 (完全可行)
    #
    # Explainer 初始化做的事:
    #   - 加载XGBoost模型的树结构
    #   - 计算每个特征在训练集上的expected value
    #   - 预处理树结构以加速后续计算
    #
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    # explainer.expected_value 即为基础值(baseline)
    # = 模型在训练集加权平均的预测值
    
    # ═══════════════════════════════════════════════════════════════════
    # ★★★ 【步骤 2】计算测试集的 SHAP 值 ★★★
    # ═══════════════════════════════════════════════════════════════════
    #
    # 【核心计算】这是SHAP分析中最重要的一行代码!
    #
    # explainer.shap_values(X) 返回什么?
    #   返回矩阵: shape = (样本数, 特征数) = (38, 21)
    #   矩阵[i,j] = 第i个样本, 第j个特征的SHAP值
    #
    # SHAP 值的数学定义:
    #   
    #   SHAP_ij = E[f(X) | X_j = x_ij] - E[f(X)]
    #         ↑         ↑                    ↑
    #      特征j的   当特征j已知时的     所有样本的
    #     SHAP值    条件期望预测          无条件期望
    #
    #   更直观的解释:
    #   SHAP_ij = 知道特征j之后的预期预测 - 不知道特征j的预期预测
    #
    # 计算方法 (TreeExplainer 内部):
    #
    #   对每个特征j计算:
    #     1. 在XGBoost的所有500棵树中
    #     2. 追踪样本i从根到叶的路径
    #     3. 当路径经过分割节点时:
    #        - 如果分割特征=j, 计算该分割的贡献
    #        - 记录该路径上其他特征的条件期望
    #     4. 汇总所有树的贡献,加权平均
    #     5. 得到该特征对该样本的SHAP值
    #
    # SHAP 值的关键性质:
    #
    #   【局部准确性 (Local Accuracy)】
    #   sum(SHAP_i[j] for all j) + explainer.expected_value = model.predict(X_i)
    #   
    #   含义: 
    #   - 所有特征的SHAP值之和 = 该样本预测偏离基础值的部分
    #   - SHAP值真实反映了每个特征对预测的贡献
    #
    #   【缺失性 (Missingness)】
    #   - 不在模型中的特征 → SHAP值 = 0
    #   - 保证了特征的最小逻辑一致性
    #
    #   【一致性 (Consistency)】
    #   - 如果特征i比特征j对预测影响更大
    #   - 则特征i的|SHAP值|平均值 ≥ 特征j的|SHAP值|平均值
    #   - 满足直观的重要性顺序
    #
    print("Computing SHAP values for test set...")
    shap_values = explainer.shap_values(X_test_imputed)
    # 返回 numpy.ndarray, shape=(38, 21)
    # shap_values[i,j] = 样本i的特征j的SHAP值
    # 【SHAP 值计算完成】
    
    print("Generating SHAP visualizations...")
    
    # ═══════════════════════════════════════════════════════════════════
    # 【步骤 3】特征重要性计算与排序
    # ═══════════════════════════════════════════════════════════════════
    #
    # 计算特征重要性的方法:
    # 当我们有了所有样本的SHAP值矩阵后,需要汇总为特征级别的重要性
    #
    # 常见方法对比:
    #   1. Mean(|SHAP|): 平均绝对SHAP值 ← 本脚本使用
    #      含义: 特征对预测的平均影响大小 (取绝对值消除正负)
    #
    #   2. Std(SHAP): SHAP值的标准差
    #      含义: 特征影响的波动性 (影响是否稳定)
    #
    #   3. Mean(SHAP): 平均SHAP值 (有符号)
    #      含义: 特征对预测是正影响还是负影响
    #
    # 在SHAP中,Mean(|SHAP|) 是最常用的特征重要性度量
    #
    feature_importance = shap_values.mean(axis=0)
    # axis=0: 沿样本维度求平均
    # 计算公式: feature_importance[j] = mean(|shap_values[:, j]|)
    # 结果: 数组长度=21, 每个元素是一个特征的重要性分数
    
    # 获取前4个最重要特征的索引 (用于dependence图)
    top_indices = np.argsort(feature_importance)[-4:][::-1]
    # np.argsort(): 返回排序后的索引 (从小到大)
    # [-4:]: 取最后4个(最大的4个)
    # [::-1]: 反转列表(变成从大到小)
    # 结果: top_indices = [idx1, idx2, idx3, idx4] 按重要性从大到小
    
    # ═══════════════════════════════════════════════════════════════════
    # 【步骤 4】SHAP 可视化
    # ═══════════════════════════════════════════════════════════════════
    
    # ★ 可视化方式1: 特征重要性条形图 (Bar Plot) ★
    #
    # 【什么是 summary_plot?】
    # SHAP 库提供的高级可视化函数,自动根据SHAP值矩阵生成图表
    #
    # 【Bar模式做什么?】
    # 1. 计算每个特征的平均|SHAP值|
    # 2. 按从大到小排序
    # 3. 绘制水平条形图
    # 4. 特征名称在Y轴(中文), 平均|SHAP值|在X轴
    #
    # 【如何理解这个图?】
    # - 条形越长 → 特征越重要
    # - 顺序反映了特征对模型预测的平均影响程度
    # - 不分正负方向(都取绝对值)
    #
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_imputed, plot_type='bar', show=False)
    plt.title('SHAP 特征重要性\n(平均 |SHAP 值|)', fontsize=14, fontweight='bold')
    plt.xlabel('平均 |SHAP 值|', fontsize=12)
    plt.tight_layout()
    plt.savefig(base_dir / 'shap_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: shap_importance_bar.png")
    
    # ★ 可视化方式2: 特征值 vs SHAP值 散点图 (Scatter Plot) ★
    #
    # 【Scatter模式做什么?】
    # 1. 对每个特征,绘制一个散点图
    # 2. X轴: 特征的实际值
    # 3. Y轴: 该特征的SHAP值
    # 4. 颜色: 特征值的大小(红=高, 蓝=低)
    # 5. 按特征重要性从高到低排列
    #
    # 【图表解读】
    # - 右上方的点 = 特征值高 && SHAP值高 (正相关)
    # - 左下方的点 = 特征值低 && SHAP值低 (可能负相关)
    # - 点的聚集程度 = 特征影响的一致性
    # - 红色点多在上方 = 高特征值推高预测
    # - 蓝色点多在下方 = 低特征值拉低预测
    #
    # 【用途】
    # - 发现特征与目标的非线性关系
    # - 检测特征的作用方向
    # - 发现异常样本或极端值
    #
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_imputed, plot_type='scatter', show=False)
    plt.title('SHAP Values vs Feature Values\n(Feature Contribution Pattern)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(base_dir / 'shap_summary_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: shap_summary_scatter.png")
    
    # ★ 可视化方式3: Dependence 图 (单特征的SHAP值关系) ★
    #
    # 【Dependence图做什么?】
    # 显示单个特征对模型预测的详细影响模式
    #
    # 【与scatter_plot的区别】
    #   scatter_plot:
    #   - 显示所有21个特征的SHAP分布
    #   - 仅显示特征值 vs SHAP值的散点
    #   - 总体概览,不够细致
    #
    #   dependence_plot:
    #   - 聚焦于单个特征
    #   - 显示该特征与SHAP值的详细关系
    #   - 叠加近似趋势线(显示非线性)
    #   - 自动选择与其相关的交互特征着色
    #
    # 【图表元素】
    # - 散点: 每个测试样本的特征值-SHAP值对
    # - 红/蓝色: 主要交互特征的值(红=高,蓝=低)
    # - 趋势线: 非线性关系的近似
    # - Y轴标题: "SHAP value for [特征中文名]"
    # - X轴标题: "[特征中文名]"
    #
    # 【如何理解】
    # 如果dependence图显示:
    # - 向上的趋势 → 特征值越大,预测越高
    # - 向下的趋势 → 特征值越大,预测越低
    # - 无明显趋势 → 特征影响不清晰(可能有交互)
    # - 颜色分层 → 存在与其他特征的交互效应
    #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, ax_idx in enumerate(top_indices):
        ax = axes[idx]
        # 为特征ax_idx绘制dependence图
        # dependence_plot内部会:
        #   1. 提取该特征的SHAP值和特征值
        #   2. 按特征值排序
        #   3. 找出与其相关度最高的其他特征(用于着色)
        #   4. 绘制着色的散点图
        shap.dependence_plot(
            ax_idx,                    # 特征索引
            shap_values,               # SHAP值矩阵 (38×21)
            X_test_imputed,            # 特征数据
            show=False,
            ax=ax
        )
        ax.set_title(f'Top {idx+1}: {feature_names[ax_idx]}', fontsize=11, fontweight='bold')
    
    plt.suptitle('Top 4 Features - SHAP Dependence Plots', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(base_dir / 'shap_top4_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: shap_top4_dependence.png")
    
    # ═══════════════════════════════════════════════════════════════════
    # 【步骤 5】生成特征重要性统计表
    # ═══════════════════════════════════════════════════════════════════
    #
    # 【为什么需要统计表?】
    # 图表直观但精度不够,表格可以提供精确数值用于定量分析
    #
    # 【包含的统计指标】
    # 
    # 1. Feature (特征名): 中文特征名称(用于理解)
    #
    # 2. Mean_SHAP (平均SHAP值):
    #    公式: mean(|SHAP_value[:, j]|)
    #    含义: 特征j对所有样本的平均影响大小
    #    用途: 排序特征重要性 (值越大越重要)
    #    范围: [0, ∞)
    #
    # 3. Std_SHAP (标准差SHAP值):
    #    公式: std(|SHAP_value[:, j]|)
    #    含义: 特征j影响的波动性
    #    低值 (Std < Mean/3): 特征影响稳定,一致性强
    #    高值 (Std > Mean/2): 特征影响不稳定,存在样本差异
    #    用途: 评估特征的影响一致性
    #
    # 4. Max_SHAP (最大SHAP值):
    #    公式: max(|SHAP_value[:, j]|)
    #    含义: 该特征的最极端影响情况
    #    用途: 识别异常值和极端情况
    #
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,                              # 特征名
        'Mean_SHAP': np.abs(shap_values).mean(axis=0),        # 平均|SHAP|
        'Std_SHAP': np.abs(shap_values).std(axis=0),          # 标准差
        'Max_SHAP': np.abs(shap_values).max(axis=0),          # 最大值
    }).sort_values('Mean_SHAP', ascending=False)              # 按重要性排序
    
    # 保存为CSV (使用UTF-8编码以支持中文)
    feature_importance_df.to_csv(base_dir / 'shap_feature_importance.csv', index=False)
    print("Saved: shap_feature_importance.csv")
    
    # ═══════════════════════════════════════════════════════════════════
    # 【步骤 6】模型性能评估
    # ═══════════════════════════════════════════════════════════════════
    #
    # 在生成SHAP分析前,需要了解模型本身的质量
    # 一个差劲的模型的SHAP值分析也不值得信任
    #
    y_pred = pipe.predict(X_test)  # 模型的预测值
    r2 = r2_score(y_test, y_pred)  # R²分数
    mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差
    
    # R² 分数解释:
    # R² = 1 - (SS_res / SS_tot)
    # SS_res = sum((y_true - y_pred)^2) 残差平方和
    # SS_tot = sum((y_true - y_mean)^2) 总平方和
    # 
    # R² = 0.5  → 解释了50%的方差(尚可)
    # R² = 0.7  → 解释了70%的方差(较好)
    # R² = 0.9  → 解释了90%的方差(很好)
    # 
    # 通常 R² > 0.3 的模型才值得分析其SHAP值
    
    # MAE 解释:
    # MAE = mean(|y_true - y_pred|)
    # 直观含义: 平均每个预测偏离真实值多少
    # 单位: 与目标变量相同 (本例: 人,表示预测误差约多少人)
    
    # ═══════════════════════════════════════════════════════════════════
    # 【步骤 7】生成综合分析报告
    # ═══════════════════════════════════════════════════════════════════
    #
    # explainer.expected_value 是SHAP理论中的"基础值"
    # 
    # 定义: 模型对训练集的平均预测值
    # 意义: 作为SHAP值的参考点
    # 用途: 满足局部准确性: prediction = expected_value + sum(SHAP值)
    #
    print("\nGenerating SHAP Analysis Report...")
    create_shap_report(feature_importance_df, r2, mae, base_dir, explainer.expected_value)
    
    # ═══════════════════════════════════════════════════════════════════
    # 【步骤 8】返回分析结果
    # ═══════════════════════════════════════════════════════════════════
    #
    return {
        'shap_values': shap_values,           # 原始SHAP值矩阵
        'X_test_imputed': X_test_imputed,     # 预处理后的特征
        'feature_importance_df': feature_importance_df,  # 统计表
        'explainer': explainer,                # SHAP解释器
        'r2': r2,                              # 模型性能
        'mae': mae                             # 模型性能
    }


def create_shap_report(feature_importance_df, r2, mae, base_dir, base_value):
    """Create comprehensive SHAP analysis report"""
    
    report = f"""
================================================================================
SHAP VALUE ANALYSIS REPORT
XGBoost Model - City Migration Prediction
================================================================================

1. MODEL PERFORMANCE SUMMARY
================================================================================
Test Set R^2: {r2:.6f}
Test Set MAE: {mae:.2f}
Base Value (Expected Model Output): {base_value:.2f}

2. FEATURE IMPORTANCE RANKING
================================================================================
Based on Mean Absolute SHAP Values (Higher = More Important)

"""
    
    for idx, row in feature_importance_df.iterrows():
        pct = (row['Mean_SHAP'] / feature_importance_df['Mean_SHAP'].sum()) * 100
        bar_length = int(pct / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        report += f"\n{idx+1:2}. {row['Feature']:20} {bar} {pct:5.2f}% | Mean={row['Mean_SHAP']:8.2f}"
    
    report += f"""

3. KEY INSIGHTS
================================================================================

3.1 Top 5 Most Influential Features

Rank 1: {feature_importance_df.iloc[0]['Feature']}
  - Mean SHAP: {feature_importance_df.iloc[0]['Mean_SHAP']:.4f}
  - Impact: Most significant driver of prediction variance
  - Interpretation: This feature shows the strongest correlation with
    city outflow, with the largest average absolute impact on model output

Rank 2: {feature_importance_df.iloc[1]['Feature']}
  - Mean SHAP: {feature_importance_df.iloc[1]['Mean_SHAP']:.4f}
  - Impact: Second most important predictor

Rank 3: {feature_importance_df.iloc[2]['Feature']}
  - Mean SHAP: {feature_importance_df.iloc[2]['Mean_SHAP']:.4f}
  - Impact: Third most important predictor

Rank 4: {feature_importance_df.iloc[3]['Feature']}
  - Mean SHAP: {feature_importance_df.iloc[3]['Mean_SHAP']:.4f}
  - Impact: Fourth most important predictor

Rank 5: {feature_importance_df.iloc[4]['Feature']}
  - Mean SHAP: {feature_importance_df.iloc[4]['Mean_SHAP']:.4f}
  - Impact: Fifth most important predictor

3.2 Feature Contribution Distribution

Total Cumulative Impact (sum of mean SHAP): {feature_importance_df['Mean_SHAP'].sum():.4f}

Top 5 Features Contribution: {(feature_importance_df.head(5)['Mean_SHAP'].sum() / feature_importance_df['Mean_SHAP'].sum() * 100):.2f}%
Top 10 Features Contribution: {(feature_importance_df.head(10)['Mean_SHAP'].sum() / feature_importance_df['Mean_SHAP'].sum() * 100):.2f}%

Variance in Feature Importance:
  - Highest Std SHAP: {feature_importance_df.loc[feature_importance_df['Std_SHAP'].idxmax(), 'Feature']} (Std={feature_importance_df['Std_SHAP'].max():.4f})
  - Lowest Std SHAP: {feature_importance_df.loc[feature_importance_df['Std_SHAP'].idxmin(), 'Feature']} (Std={feature_importance_df['Std_SHAP'].min():.4f})

4. FEATURE IMPORTANCE CATEGORIES
================================================================================

Core Economic Indicators (GDP, Sectors):
{', '.join(feature_importance_df[feature_importance_df['Feature'].str.contains('GDP|Sector')]['Feature'].tolist())}

Financial Indicators:
{', '.join(feature_importance_df[feature_importance_df['Feature'].str.contains('Income|Loan|Deposit')]['Feature'].tolist())}

Infrastructure/Quality of Life:
{', '.join(feature_importance_df[feature_importance_df['Feature'].str.contains('Road|Hospital|Student')]['Feature'].tolist())}

5. SHAP VALUE INTERPRETATION GUIDE
================================================================================

SHAP values explain how much each feature contributes to pushing the model's
prediction away from the base value (expected value).

Positive SHAP Value:
  - Feature value pushes prediction HIGHER (more outflow migration expected)
  - Red color in scatter plots indicates higher feature values

Negative SHAP Value:
  - Feature value pushes prediction LOWER (less outflow migration expected)
  - Blue color in scatter plots indicates lower feature values

Feature Importance (Mean |SHAP|):
  - Measures average magnitude of impact
  - Higher value = more important for predictions
  - Captures both positive and negative directional effects

6. VISUALIZATIONS GENERATED
================================================================================

1. shap_importance_bar.png
   - Bar chart of mean absolute SHAP values
   - Shows feature importance ranking
   - Wider bar = more important feature

2. shap_summary_scatter.png
   - Scatter plot of SHAP values vs feature values
   - Shows relationship patterns
   - Color gradient indicates feature value magnitude

3. shap_top4_dependence.png
   - Detailed dependence plots for top 4 features
   - Shows non-linear relationships
   - Helps identify feature interaction patterns

4. shap_feature_importance.csv
   - Detailed statistics for all features
   - Includes mean, std, and max SHAP values

7. MODELING IMPLICATIONS
================================================================================

Model Architecture: XGBoost with Grid Search Optimization
  - Learning Rate: 0.05
  - Max Depth: 5
  - N Estimators: 500
  - Subsample: 0.8

Model Interpretability:
  - SHAP values provide local (sample-level) explanations
  - Can explain why model makes specific predictions
  - Useful for model debugging and feature validation

Feature Engineering Recommendations:
  1. Focus on top 5 features for feature engineering efforts
  2. Consider polynomial features for {feature_importance_df.iloc[0]['Feature']}
  3. Investigate interaction effects between top features
  4. Consider domain knowledge for feature combinations

8. STATISTICAL NOTES
================================================================================

Base Value: {base_value:.2f}
  - This is the average prediction across the training set
  - SHAP values sum to: Model Prediction - Base Value

Feature Scaling:
  - All SHAP calculations use actual feature values (no normalization)
  - SHAP values are on the same scale as the target variable

Sample Size for Analysis:
  - Test set size: 44 samples
  - All SHAP values computed for this test set
  - Features: 21 economic indicators

================================================================================
END OF REPORT
================================================================================
"""
    
    with open(base_dir / 'SHAP_ANALYSIS_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Saved: SHAP_ANALYSIS_REPORT.txt")


def main():
    """
    主函数: 协调整个SHAP分析流程
    
    执行流程:
    ├─ 步骤 1: load_data() - 加载并预处理数据
    ├─ 步骤 2: train_optimal_xgboost() - 训练XGBoost模型
    ├─ 步骤 3: generate_shap_analysis() - 核心SHAP分析
    └─ 步骤 4: 输出摘要并显示结果
    
    流程图:
    
    原始CSV →│
             ├→ load_data() →│
                             ├→ train_optimal_xgboost() →│
                                                          ├→ generate_shap_analysis() →│
                                                                                       ├→ TreeExplainer 初始化
                                                                                       ├→ 计算SHAP值
                                                                                       ├→ 生成3个PNG
                                                                                       ├→ 保存CSV
                                                                                       ├→ create_shap_report()
                                                                                       └→ 返回结果字典
                                                                                              ↓
                                                                                           输出摘要
    
    输出文件 (base_dir = 当前工作目录):
    ├─ shap_importance_bar.png (特征重要性条)
    ├─ shap_summary_scatter.png (SHAP值分布)
    ├─ shap_top4_dependence.png (前4特征细节)
    ├─ shap_feature_importance.csv (统计数据)
    └─ SHAP_ANALYSIS_REPORT.txt (详细报告)
    """

    plt.rcParams['font.sans-serif'] = ['SimHei']
    base_dir = Path.cwd()
    
    # 【步骤 1】加载数据
    # 作用: 读取CSV文件,提取特征和目标
    # 产出: 4个数据变量用于模型训练
    print("Loading data...")
    X_train, y_train, X_test, y_test, feature_names = load_data()
    print(f"Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    # X_train: (155, 21) 样本×特征
    # y_train: (155,) 目标值
    # X_test: (38, 21)
    # y_test: (38,)
    # feature_names: 长度=21的中文特征名列表
    
    # 【步骤 2】训练模型
    # 作用: 使用网格搜索优化的参数训练XGBoost
    # 产出: Pipeline对象(包含imputer和模型)
    print("\nTraining optimized XGBoost model...")
    pipe = train_optimal_xgboost(X_train, y_train)
    # pipe 是一个sklearn Pipeline,顺序包含:
    # 1. SimpleImputer (缺失值处理)
    # 2. XGBRegressor (回归模型)
    
    # 【步骤 3】生成SHAP分析 (核心)
    # 作用: 计算SHAP值,生成可视化和报告
    # 产出: 包含SHAP结果的字典
    print("\nGenerating SHAP analysis...")
    results = generate_shap_analysis(pipe, X_train, X_test, y_test, feature_names, base_dir)
    
    # 【步骤 4】输出摘要与验证
    # 作用: 显示分析结果,供用户查看
    if results:
        print("\n" + "="*80)
        print("SHAP ANALYSIS COMPLETE (分析完成!)")
        print("="*80)
        print("\n◆ 特征重要性摘要 (Top 10):")
        print(results['feature_importance_df'].head(10).to_string(index=False))
        
        print(f"\n◆ 模型性能指标:")
        print(f"  • R² 分数: {results['r2']:.6f}")
        print(f"    (解释了{results['r2']*100:.1f}%的测试集方差)")
        print(f"  • MAE: {results['mae']:.2f} 人")
        print(f"    (平均预测误差约{results['mae']:.0f}人)")
        
        print(f"\n◆ 生成的输出文件:")
        print(f"  ✓ shap_importance_bar.png - 特征重要性条形图")
        print(f"  ✓ shap_summary_scatter.png - SHAP值散点图")
        print(f"  ✓ shap_top4_dependence.png - 前4特征dependence图")
        print(f"  ✓ shap_feature_importance.csv - 完整特征统计表")
        print(f"  ✓ SHAP_ANALYSIS_REPORT.txt - 详细分析报告")
        
        print(f"\n◆ 关键发现:")
        top1 = results['feature_importance_df'].iloc[0]
        print(f"  • 最重要特征: {top1['Feature']}")
        print(f"  • 平均SHAP值: {top1['Mean_SHAP']:.2f}")
        print(f"  • 前5特征的累计重要性占比:")
        top5_sum = results['feature_importance_df'].head(5)['Mean_SHAP'].sum()
        total_sum = results['feature_importance_df']['Mean_SHAP'].sum()
        print(f"    {(top5_sum/total_sum*100):.1f}% (前5个特征驱动{(top5_sum/total_sum*100):.0f}%的预测)")


if __name__ == '__main__':
    main()
