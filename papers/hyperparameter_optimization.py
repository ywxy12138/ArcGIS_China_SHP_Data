from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 超参数优化脚本（带流程化注释）
#
# 目标：在同一套训练/测试划分上，对 XGBoost / RandomForest / LightGBM
# 进行可复现的超参数搜索，并输出可直接对比的 R2/MAE 指标。
#
# 设计要点：
# 1) 统一预处理：缺失值都走中位数填补，避免不同模型因预处理差异造成偏差。
# 2) 统一评估：同一交叉验证与同一评分函数，比较更公平。
# 3) 稳定优先：Windows 环境下降低多层并行风险，保证能稳定跑完。
# 4) 可扩展：支持 log1p 目标变换和 Optuna，但默认给出更稳妥配置。
# =============================================================================

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# 说明：人口流出目标值通常右偏，log1p 能缓解长尾对模型训练的干扰。
# 为避免在当前数据上过度约束，默认关闭，可按需改成 True。
USE_LOG_TARGET = False

# Windows + 多层并行（模型并行 + CV并行）在本环境容易出现进程中断。
# 这里采用“稳定优先”的并行策略，确保能完整跑完并复现结果。
MODEL_N_JOBS = 1
SEARCH_N_JOBS = 1

# 为了兼顾结果稳定性和可运行时间，控制搜索与重复验证的规模。
# 这样仍然保留重复K折带来的稳健性，但不会把一轮优化拖得过长。
N_SPLITS = 5
N_REPEATS = 1
N_RANDOM_SEARCH_ITERS = 10
N_OPTUNA_TRIALS = 10

# Optuna 的贝叶斯搜索在本环境里代价偏高；先关闭，优先保证这一轮能完整产出结果。
RUN_OPTUNA = False


def load_data():
    """
    读取训练集/测试集，并完成最基础的数据清洗。

    处理流程：
    1) 从当前工作目录读取分层划分后的训练和测试数据。
    2) 固定使用 21 个经济指标作为特征列，确保不同模型输入一致。
    3) 使用 to_numeric(errors='coerce') 将异常字符串转为 NaN，避免类型报错。
    4) 仅按目标列是否缺失进行样本过滤，防止无标签样本干扰训练。

    返回：X_train, y_train, X_test, y_test
    """
    base_dir = Path.cwd()
    train_df = pd.read_csv(base_dir / 'city_data_train_stratified_8_2.csv')
    test_df = pd.read_csv(base_dir / 'city_data_test_stratified_8_2.csv')
    
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
    target_col = '2024年总流出人口数'
    
    X_train = train_df[raw_feature_cols].apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(train_df[target_col], errors='coerce')
    X_test = test_df[raw_feature_cols].apply(pd.to_numeric, errors='coerce')
    y_test = pd.to_numeric(test_df[target_col], errors='coerce')
    
    # 仅保留目标值非空样本：监督学习必须保证 y 可用。
    train_mask = y_train.notna()
    test_mask = y_test.notna()
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_test = X_test.loc[test_mask].copy()
    y_test = y_test.loc[test_mask].copy()
    
    return X_train, y_train, X_test, y_test


def build_estimator(model):
    """
    统一构建估计器：
    1) 先做中位数填补，提升对缺失值的鲁棒性
    2) 可选对目标做 log1p/expm1 变换，降低极端值对损失函数的主导
    """
    # 统一建模管道：先填补缺失，再训练模型。
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', model)
    ])

    # 对目标值做 log1p 的作用：
    # - 缓解右偏/长尾，降低极端样本对平方误差的主导。
    # - 常见于人口、金额类预测任务。
    # 预测输出通过 expm1 自动反变换回原量纲，便于解释 MAE/R2。
    if USE_LOG_TARGET:
        return TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    return pipe


def with_prefix(params):
    """
    根据估计器结构自动补参数前缀。
    使用目标变换后，模型位于 TransformedTargetRegressor -> Pipeline -> model。
    """
    # sklearn 在嵌套估计器中使用“前缀__参数名”定位超参数。
    # 这里统一处理两种结构，避免手工拼接出错：
    # - 不做目标变换：Pipeline -> model__*
    # - 做目标变换：TTR -> regressor(Pipeline) -> regressor__model__*
    prefix = 'regressor__model__' if USE_LOG_TARGET else 'model__'
    return {f'{prefix}{k}': v for k, v in params.items()}


def optimize_xgboost_random(X_train, y_train, cv):
    """
    XGBoost 随机搜索：
    - 通过较宽参数空间先做“粗搜索”，快速覆盖更多组合。
    - 重点搜索学习率、树深、采样比例与正则项，提升泛化能力。
    """
    param_dist = with_prefix({
        'n_estimators': [300, 500, 800, 1000],
        'learning_rate': [0.01, 0.03, 0.05, 0.08],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5, 8],
        'gamma': [0.0, 0.5, 1.0],
        'reg_alpha': [0.0, 0.5, 1.0],
        'reg_lambda': [1.0, 3.0, 5.0],
    })

    estimator = build_estimator(
        XGBRegressor(random_state=42, n_jobs=MODEL_N_JOBS, objective='reg:squarederror', tree_method='hist')
    )

    # RandomizedSearchCV 的优势：同等预算下通常比小网格更容易摸到优解区域。
    search = RandomizedSearchCV(estimator, param_dist, n_iter=N_RANDOM_SEARCH_ITERS, cv=cv,
                                scoring='r2', n_jobs=SEARCH_N_JOBS, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def optimize_xgboost_grid(X_train, y_train, cv):
    """
    XGBoost 网格搜索：
    - 在经验较优区间做“精搜索”，用于和随机搜索形成互补。
    - 网格规模受控，避免运行时间不可控。
    """
    param_grid = with_prefix({
        'n_estimators': [500, 800],
        'learning_rate': [0.03, 0.05],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    })

    estimator = build_estimator(
        XGBRegressor(random_state=42, n_jobs=MODEL_N_JOBS, objective='reg:squarederror', tree_method='hist')
    )

    search = GridSearchCV(estimator, param_grid, cv=cv,
                          scoring='r2', n_jobs=SEARCH_N_JOBS)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def optimize_xgboost_optuna(X_train, y_train, cv):
    """
    XGBoost Optuna（贝叶斯优化）：
    - 通过历史试验结果引导下一步采样，通常比纯随机更高效。
    - 当前脚本默认关闭 RUN_OPTUNA，仅在需要更深搜索时开启。
    """
    if not OPTUNA_AVAILABLE:
        return None, None, None
    
    def objective(trial):
        # trial.suggest_* 定义搜索空间；log=True 适合学习率这类尺度敏感参数。
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42,
            'n_jobs': MODEL_N_JOBS,
        }

        estimator = build_estimator(XGBRegressor(**params, objective='reg:squarederror', tree_method='hist'))
        # 目标函数使用交叉验证均值 R2，降低单次切分偶然性。
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='r2', n_jobs=SEARCH_N_JOBS)
        return float(np.mean(scores))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    best_params['n_jobs'] = MODEL_N_JOBS
    
    estimator = build_estimator(XGBRegressor(**best_params, objective='reg:squarederror', tree_method='hist'))
    estimator.fit(X_train, y_train)
    scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='r2', n_jobs=SEARCH_N_JOBS)
    return estimator, best_params, float(np.mean(scores))


def optimize_rf_random(X_train, y_train, cv):
    """
    RandomForest 随机搜索：
    - 重点调 n_estimators / 深度 / 划分阈值 / max_features。
    - 通过控制树复杂度抑制过拟合，提升测试集拟合稳定性。
    """
    param_dist = with_prefix({
        'n_estimators': [400, 600, 900, 1200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.8],
    })

    estimator = build_estimator(RandomForestRegressor(random_state=42, n_jobs=MODEL_N_JOBS))

    search = RandomizedSearchCV(estimator, param_dist, n_iter=N_RANDOM_SEARCH_ITERS, cv=cv,
                                scoring='r2', n_jobs=SEARCH_N_JOBS, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def optimize_rf_grid(X_train, y_train, cv):
    """
    RandomForest 网格搜索：
    - 在常见有效参数上做小范围精调，补充随机搜索。
    """
    param_grid = with_prefix({
        'n_estimators': [600, 900],
        'max_depth': [15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    })

    estimator = build_estimator(RandomForestRegressor(random_state=42, n_jobs=MODEL_N_JOBS))

    search = GridSearchCV(estimator, param_grid, cv=cv,
                          scoring='r2', n_jobs=SEARCH_N_JOBS)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def optimize_rf_optuna(X_train, y_train, cv):
    """
    RandomForest Optuna：
    - 适合在随机/网格后继续挖掘更优组合。
    - with_depth 用于同时探索“有限深度”和“无限深度”两种结构。
    """
    if not OPTUNA_AVAILABLE:
        return None, None, None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 30) if trial.suggest_categorical('with_depth', [True, False]) else None,
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42,
            'n_jobs': MODEL_N_JOBS,
        }
        
        estimator = build_estimator(RandomForestRegressor(**params))
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='r2', n_jobs=SEARCH_N_JOBS)
        return float(np.mean(scores))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    best_params['n_jobs'] = MODEL_N_JOBS
    
    # Optuna 中的辅助开关参数不属于模型原生参数，这里做收尾清理。
    if 'with_depth' in best_params:
        if not best_params.pop('with_depth'):
            best_params['max_depth'] = None
    
    estimator = build_estimator(RandomForestRegressor(**best_params))
    estimator.fit(X_train, y_train)
    scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='r2', n_jobs=SEARCH_N_JOBS)
    return estimator, best_params, float(np.mean(scores))


def optimize_lgbm_random(X_train, y_train, cv):
    """
    LightGBM 随机搜索：
    - 调整叶子数、学习率、采样比例与正则项。
    - 对中小样本表格数据常有较好拟合效率和精度平衡。
    """
    param_dist = with_prefix({
        'n_estimators': [500, 800, 1000],
        'learning_rate': [0.01, 0.03, 0.05, 0.08],
        'num_leaves': [20, 31, 50, 70],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.001, 0.01, 0.1],
        'reg_alpha': [0.0, 0.5, 1.0],
        'reg_lambda': [0.0, 1.0, 3.0],
    })

    estimator = build_estimator(LGBMRegressor(random_state=42, n_jobs=MODEL_N_JOBS, verbosity=-1))

    search = RandomizedSearchCV(estimator, param_dist, n_iter=N_RANDOM_SEARCH_ITERS, cv=cv,
                                scoring='r2', n_jobs=SEARCH_N_JOBS, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def optimize_lgbm_grid(X_train, y_train, cv):
    """
    LightGBM 网格搜索：
    - 在较优区间继续微调，通常能进一步压低 MAE。
    """
    param_grid = with_prefix({
        'n_estimators': [700, 1000],
        'learning_rate': [0.02, 0.05],
        'num_leaves': [31, 50, 70],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    })

    estimator = build_estimator(LGBMRegressor(random_state=42, n_jobs=MODEL_N_JOBS, verbosity=-1))

    search = GridSearchCV(estimator, param_grid, cv=cv,
                          scoring='r2', n_jobs=SEARCH_N_JOBS)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def optimize_lgbm_optuna(X_train, y_train, cv):
    """
    LightGBM Optuna：
    - 用贝叶斯优化进一步探索连续参数空间。
    - 默认关闭，按计算预算选择性开启。
    """
    if not OPTUNA_AVAILABLE:
        return None, None, None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 900),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.0001, 0.1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42,
            'n_jobs': MODEL_N_JOBS,
            'verbosity': -1,
        }
        
        estimator = build_estimator(LGBMRegressor(**params))
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='r2', n_jobs=SEARCH_N_JOBS)
        return float(np.mean(scores))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    best_params['n_jobs'] = MODEL_N_JOBS
    best_params['verbosity'] = -1
    
    estimator = build_estimator(LGBMRegressor(**best_params))
    estimator.fit(X_train, y_train)
    scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='r2', n_jobs=SEARCH_N_JOBS)
    return estimator, best_params, float(np.mean(scores))


def main():
    """
    主流程：
    1) 读取数据
    2) 设定统一交叉验证策略
    3) 依次运行三类模型的随机搜索与网格搜索（可选 Optuna）
    4) 在测试集上统一计算 Test_R2/Test_MAE
    5) 汇总排序并保存到 optimization_results.csv
    """
    print('Loading data...')
    X_train, y_train, X_test, y_test = load_data()
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    # RepeatedKFold 比单次 KFold 更稳健，可降低单次划分偶然性对超参选择的影响。
    # RepeatedKFold 的意义：相比单次 KFold，重复采样能降低偶然划分带来的方差。
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    print(f'Cross-validation: RepeatedKFold({N_SPLITS} folds x {N_REPEATS} repeats), log-target={USE_LOG_TARGET}')

    results = []
    
    print('\n' + '='*80)
    print('XGBoost Optimization')
    print('='*80)
    
    print('Random Search...')
    try:
        # 先随机搜索快速定位较优区域，再看是否需要更细粒度精调。
        model_rs, params_rs, score_rs = optimize_xgboost_random(X_train, y_train, cv)
        model_rs.fit(X_train, y_train)
        pred_rs = model_rs.predict(X_test)
        r2_rs = r2_score(y_test, pred_rs)
        mae_rs = mean_absolute_error(y_test, pred_rs)
        # 统一记录字段，便于后续做模型横向对比与可视化。
        results.append({'Model': 'XGBoost', 'Method': 'Random', 'TargetTransform': 'log1p' if USE_LOG_TARGET else 'none', 'CV_R2': score_rs, 'Test_R2': r2_rs, 'Test_MAE': mae_rs})
        print(f'XGBoost Random: CV_R2={score_rs:.6f}, Test_R2={r2_rs:.6f}, Test_MAE={mae_rs:.2f}')
    except Exception as e:
        print(f'Error: {e}')
    
    print('Grid Search...')
    try:
        model_gs, params_gs, score_gs = optimize_xgboost_grid(X_train, y_train, cv)
        model_gs.fit(X_train, y_train)
        pred_gs = model_gs.predict(X_test)
        r2_gs = r2_score(y_test, pred_gs)
        mae_gs = mean_absolute_error(y_test, pred_gs)
        results.append({'Model': 'XGBoost', 'Method': 'Grid', 'TargetTransform': 'log1p' if USE_LOG_TARGET else 'none', 'CV_R2': score_gs, 'Test_R2': r2_gs, 'Test_MAE': mae_gs})
        print(f'XGBoost Grid: CV_R2={score_gs:.6f}, Test_R2={r2_gs:.6f}, Test_MAE={mae_gs:.2f}')
    except Exception as e:
        print(f'Error: {e}')
    
    if OPTUNA_AVAILABLE and RUN_OPTUNA:
        print('Optuna Bayesian...')
        try:
            model_opt, params_opt, score_opt = optimize_xgboost_optuna(X_train, y_train, cv)
            if model_opt is not None:
                pred_opt = model_opt.predict(X_test)
                r2_opt = r2_score(y_test, pred_opt)
                mae_opt = mean_absolute_error(y_test, pred_opt)
                results.append({'Model': 'XGBoost', 'Method': 'Optuna', 'TargetTransform': 'log1p' if USE_LOG_TARGET else 'none', 'CV_R2': score_opt, 'Test_R2': r2_opt, 'Test_MAE': mae_opt})
                print(f'XGBoost Optuna: CV_R2={score_opt:.6f}, Test_R2={r2_opt:.6f}, Test_MAE={mae_opt:.2f}')
        except Exception as e:
            print(f'Error: {e}')
    
    print('\n' + '='*80)
    print('RandomForest Optimization')
    print('='*80)
    
    print('Random Search...')
    try:
        model_rs, params_rs, score_rs = optimize_rf_random(X_train, y_train, cv)
        model_rs.fit(X_train, y_train)
        pred_rs = model_rs.predict(X_test)
        r2_rs = r2_score(y_test, pred_rs)
        mae_rs = mean_absolute_error(y_test, pred_rs)
        results.append({'Model': 'RandomForest', 'Method': 'Random', 'TargetTransform': 'log1p' if USE_LOG_TARGET else 'none', 'CV_R2': score_rs, 'Test_R2': r2_rs, 'Test_MAE': mae_rs})
        print(f'RF Random: CV_R2={score_rs:.6f}, Test_R2={r2_rs:.6f}, Test_MAE={mae_rs:.2f}')
    except Exception as e:
        print(f'Error: {e}')
    
    print('Grid Search...')
    try:
        model_gs, params_gs, score_gs = optimize_rf_grid(X_train, y_train, cv)
        model_gs.fit(X_train, y_train)
        pred_gs = model_gs.predict(X_test)
        r2_gs = r2_score(y_test, pred_gs)
        mae_gs = mean_absolute_error(y_test, pred_gs)
        results.append({'Model': 'RandomForest', 'Method': 'Grid', 'TargetTransform': 'log1p' if USE_LOG_TARGET else 'none', 'CV_R2': score_gs, 'Test_R2': r2_gs, 'Test_MAE': mae_gs})
        print(f'RF Grid: CV_R2={score_gs:.6f}, Test_R2={r2_gs:.6f}, Test_MAE={mae_gs:.2f}')
    except Exception as e:
        print(f'Error: {e}')
    
    if OPTUNA_AVAILABLE and RUN_OPTUNA:
        print('Optuna Bayesian...')
        try:
            model_opt, params_opt, score_opt = optimize_rf_optuna(X_train, y_train, cv)
            if model_opt is not None:
                pred_opt = model_opt.predict(X_test)
                r2_opt = r2_score(y_test, pred_opt)
                mae_opt = mean_absolute_error(y_test, pred_opt)
                results.append({'Model': 'RandomForest', 'Method': 'Optuna', 'TargetTransform': 'log1p' if USE_LOG_TARGET else 'none', 'CV_R2': score_opt, 'Test_R2': r2_opt, 'Test_MAE': mae_opt})
                print(f'RF Optuna: CV_R2={score_opt:.6f}, Test_R2={r2_opt:.6f}, Test_MAE={mae_opt:.2f}')
        except Exception as e:
            print(f'Error: {e}')
    
    print('\n' + '='*80)
    print('LightGBM Optimization')
    print('='*80)
    
    print('Random Search...')
    try:
        model_rs, params_rs, score_rs = optimize_lgbm_random(X_train, y_train, cv)
        model_rs.fit(X_train, y_train)
        pred_rs = model_rs.predict(X_test)
        r2_rs = r2_score(y_test, pred_rs)
        mae_rs = mean_absolute_error(y_test, pred_rs)
        results.append({'Model': 'LightGBM', 'Method': 'Random', 'TargetTransform': 'log1p' if USE_LOG_TARGET else 'none', 'CV_R2': score_rs, 'Test_R2': r2_rs, 'Test_MAE': mae_rs})
        print(f'LGBM Random: CV_R2={score_rs:.6f}, Test_R2={r2_rs:.6f}, Test_MAE={mae_rs:.2f}')
    except Exception as e:
        print(f'Error: {e}')
    
    print('Grid Search...')
    try:
        model_gs, params_gs, score_gs = optimize_lgbm_grid(X_train, y_train, cv)
        model_gs.fit(X_train, y_train)
        pred_gs = model_gs.predict(X_test)
        r2_gs = r2_score(y_test, pred_gs)
        mae_gs = mean_absolute_error(y_test, pred_gs)
        results.append({'Model': 'LightGBM', 'Method': 'Grid', 'TargetTransform': 'log1p' if USE_LOG_TARGET else 'none', 'CV_R2': score_gs, 'Test_R2': r2_gs, 'Test_MAE': mae_gs})
        print(f'LGBM Grid: CV_R2={score_gs:.6f}, Test_R2={r2_gs:.6f}, Test_MAE={mae_gs:.2f}')
    except Exception as e:
        print(f'Error: {e}')
    
    if OPTUNA_AVAILABLE and RUN_OPTUNA:
        print('Optuna Bayesian...')
        try:
            model_opt, params_opt, score_opt = optimize_lgbm_optuna(X_train, y_train, cv)
            if model_opt is not None:
                pred_opt = model_opt.predict(X_test)
                r2_opt = r2_score(y_test, pred_opt)
                mae_opt = mean_absolute_error(y_test, pred_opt)
                results.append({'Model': 'LightGBM', 'Method': 'Optuna', 'TargetTransform': 'log1p' if USE_LOG_TARGET else 'none', 'CV_R2': score_opt, 'Test_R2': r2_opt, 'Test_MAE': mae_opt})
                print(f'LGBM Optuna: CV_R2={score_opt:.6f}, Test_R2={r2_opt:.6f}, Test_MAE={mae_opt:.2f}')
        except Exception as e:
            print(f'Error: {e}')
    
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)
    
    # 以测试集 R2 降序排序，优先查看泛化表现更好的方案。
    results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
    print(results_df.to_string(index=False))
    
    base_dir = Path.cwd()
    # 输出统一结果文件，供后续 SHAP/ALE 解释阶段直接读取。
    results_df.to_csv(base_dir / 'optimization_results.csv', index=False)
    print(f'\nResults saved to optimization_results.csv')


if __name__ == '__main__':
    main()
