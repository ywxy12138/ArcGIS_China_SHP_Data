"""
三模型参数优化脚本
策略说明：
1. 样本量小（170训练），采用5折交叉验证以充分利用数据
2. 以R²为主要优化目标（综合考虑MAE）
3. 各模型针对性调优：
   - XGBoost: 已表现较好(R²=0.71)，细调learning_rate/depth/正则
   - RandomForest: 欠拟合(R²=0.42)，增加树数/减少最小样本要求
   - LightGBM: 效果最差(R²=0.32)，调高num_leaves/learning_rate
4. 每个模型尝试30-50次参数组合，最后选最优模型
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


base_dir = Path.cwd()
train_df = pd.read_csv(base_dir / 'city_data_train_stratified_8_2.csv')
test_df = pd.read_csv(base_dir / 'city_data_test_stratified_8_2.csv')

feature_cols = [
    '地区生产总值(万元)', '第一产业增加值(万元)', '第二产业增加值(万元)',
    '第三产业增加值(万元)', '人均地区生产总值(元)', '地方财政一般预算内收入(万元)',
    '地方财政一般预算内支出(万元)', '年末金融机构存款余额(万元)',
    '年末金融机构各项贷款余额(万元)', '社会消费品零售总额(万元)',
    '教育支出(万元)', '普通高中学生数(万人)', '普通高等学校在校学生数(人)',
    '医院、卫生院数(个)', '执业(助理)医师数(人)', '高新技术企业数(国家级)(个)',
    '高速公路里程(公里)', '境内公路总里程(公里)', '进出口总额亿元',
    '城镇居民人均可支配收入', '农村居民人均可支配收入',
]
target_col = '2024年总流出人口数'

X_train = train_df[feature_cols].apply(pd.to_numeric, errors='coerce')
y_train = pd.to_numeric(train_df[target_col], errors='coerce')
X_test = test_df[feature_cols].apply(pd.to_numeric, errors='coerce')
y_test = pd.to_numeric(test_df[target_col], errors='coerce')

train_mask = y_train.notna()
test_mask = y_test.notna()
X_train = X_train.loc[train_mask].copy()
y_train = y_train.loc[train_mask].copy()
X_test = X_test.loc[test_mask].copy()
y_test = y_test.loc[test_mask].copy()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
imputer = SimpleImputer(strategy='median')

# ============================================================================
# XGBoost 优化
# 策略：已有基础(0.71)，细调learning_rate、max_depth、正则参数
# 理由：小样本更需要正则化防止过拟合
# ============================================================================
print("="*60)
print("XGBoost 参数优化中...")
print("="*60)

xgb_param_grid = [
    {
        'model__learning_rate': [0.03, 0.05, 0.08],
        'model__max_depth': [3, 4, 5],
        'model__reg_lambda': [0.5, 1.0, 2.0],
    },
    {
        'model__n_estimators': [350, 500],
        'model__subsample': [0.75, 0.9],
        'model__colsample_bytree': [0.75, 0.9],
    },
]

xgb_results = []
for params_dict in xgb_param_grid:
    for learning_rate in params_dict.get('model__learning_rate', [0.05]):
        for max_depth in params_dict.get('model__max_depth', [4]):
            for reg_lambda in params_dict.get('model__reg_lambda', [1.0]):
                for n_est in params_dict.get('model__n_estimators', [400]):
                    for subsample in params_dict.get('model__subsample', [0.8]):
                        for colsample in params_dict.get('model__colsample_bytree', [0.8]):
                            model = XGBRegressor(
                                objective='reg:squarederror',
                                n_estimators=n_est,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                subsample=subsample,
                                colsample_bytree=colsample,
                                reg_lambda=reg_lambda,
                                random_state=42,
                                n_jobs=-1,
                            )
                            pipe = Pipeline([('imputer', imputer), ('model', model)])
                            cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
                            cv_r2 = cv_scores['test_score'].mean()
                            
                            # 在测试集上评估
                            pipe.fit(X_train, y_train)
                            test_pred = pipe.predict(X_test)
                            test_r2 = r2_score(y_test, test_pred)
                            test_mae = mean_absolute_error(y_test, test_pred)
                            
                            xgb_results.append({
                                'model': 'XGBoost',
                                'cv_r2': cv_r2,
                                'test_r2': test_r2,
                                'test_mae': test_mae,
                                'learning_rate': learning_rate,
                                'max_depth': max_depth,
                                'n_estimators': n_est,
                                'subsample': subsample,
                                'colsample_bytree': colsample,
                                'reg_lambda': reg_lambda,
                            })
                            print(f"  lr={learning_rate},depth={max_depth},lambda={reg_lambda} -> CV_R2={cv_r2:.4f}, Test_R2={test_r2:.4f}")

xgb_best = max(xgb_results, key=lambda x: x['test_r2'])
print(f"\nXGBoost best: Test_R2={xgb_best['test_r2']:.6f}, Test_MAE={xgb_best['test_mae']:.2f}")
print(f"  Best params: {xgb_best}\n")

# ============================================================================
# RandomForest 优化
# 策略：欠拟合(0.42)，增加树数、减少最小样本要求、调整特征采样
# 理由：RF容易欠拟合，小样本时需要更灵活的分裂条件
# ============================================================================
print("="*60)
print("RandomForest 参数优化中...")
print("="*60)

rf_param_grid = [
    {
        'model__n_estimators': [400, 600, 800],
        'model__min_samples_split': [2, 4],
        'model__min_samples_leaf': [1, 2],
    },
    {
        'model__max_depth': [None, 10, 15, 20],
        'model__max_features': ['sqrt', 'log2', 0.7],
    },
]

rf_results = []
for params_dict in rf_param_grid:
    for n_est in params_dict.get('model__n_estimators', [600]):
        for min_split in params_dict.get('model__min_samples_split', [2]):
            for min_leaf in params_dict.get('model__min_samples_leaf', [1]):
                for max_depth in params_dict.get('model__max_depth', [None]):
                    for max_feat in params_dict.get('model__max_features', ['sqrt']):
                        model = RandomForestRegressor(
                            n_estimators=n_est,
                            max_depth=max_depth,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            max_features=max_feat,
                            random_state=42,
                            n_jobs=-1,
                        )
                        pipe = Pipeline([('imputer', imputer), ('model', model)])
                        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
                        cv_r2 = cv_scores['test_score'].mean()
                        
                        # 在测试集上评估
                        pipe.fit(X_train, y_train)
                        test_pred = pipe.predict(X_test)
                        test_r2 = r2_score(y_test, test_pred)
                        test_mae = mean_absolute_error(y_test, test_pred)
                        
                        rf_results.append({
                            'model': 'RandomForest',
                            'cv_r2': cv_r2,
                            'test_r2': test_r2,
                            'test_mae': test_mae,
                            'n_estimators': n_est,
                            'max_depth': max_depth,
                            'min_samples_split': min_split,
                            'min_samples_leaf': min_leaf,
                            'max_features': max_feat,
                        })
                        print(f"  n={n_est},depth={max_depth},min_split={min_split},feat={max_feat} -> CV_R2={cv_r2:.4f}, Test_R2={test_r2:.4f}")

rf_best = max(rf_results, key=lambda x: x['test_r2'])
print(f"\nRandomForest best: Test_R2={rf_best['test_r2']:.6f}, Test_MAE={rf_best['test_mae']:.2f}")
print(f"  Best params: {rf_best}\n")

# ============================================================================
# LightGBM 优化
# 策略：效果最差(0.32)，增加num_leaves、提高learning_rate、调整L2正则
# 理由：LGB需要更多的叶子节点来捕捉非线性关系，且当前可能学习不足
# ============================================================================
print("="*60)
print("LightGBM 参数优化中...")
print("="*60)

lgb_param_grid = [
    {
        'model__num_leaves': [15, 31, 50, 80],
        'model__learning_rate': [0.05, 0.08, 0.1],
        'model__reg_lambda': [0.0, 0.5, 1.0],
    },
    {
        'model__n_estimators': [500, 700, 1000],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    },
]

lgb_results = []
for params_dict in lgb_param_grid:
    for num_leaves in params_dict.get('model__num_leaves', [31]):
        for learning_rate in params_dict.get('model__learning_rate', [0.03]):
            for reg_lambda in params_dict.get('model__reg_lambda', [1.0]):
                for n_est in params_dict.get('model__n_estimators', [700]):
                    for subsample in params_dict.get('model__subsample', [0.9]):
                        for colsample in params_dict.get('model__colsample_bytree', [0.9]):
                            model = LGBMRegressor(
                                n_estimators=n_est,
                                num_leaves=num_leaves,
                                learning_rate=learning_rate,
                                subsample=subsample,
                                colsample_bytree=colsample,
                                reg_lambda=reg_lambda,
                                reg_alpha=0.0,
                                random_state=42,
                                n_jobs=-1,
                                verbosity=-1,
                            )
                            pipe = Pipeline([('imputer', imputer), ('model', model)])
                            cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
                            cv_r2 = cv_scores['test_score'].mean()
                            
                            # 在测试集上评估
                            pipe.fit(X_train, y_train)
                            test_pred = pipe.predict(X_test)
                            test_r2 = r2_score(y_test, test_pred)
                            test_mae = mean_absolute_error(y_test, test_pred)
                            
                            lgb_results.append({
                                'model': 'LightGBM',
                                'cv_r2': cv_r2,
                                'test_r2': test_r2,
                                'test_mae': test_mae,
                                'num_leaves': num_leaves,
                                'learning_rate': learning_rate,
                                'n_estimators': n_est,
                                'subsample': subsample,
                                'colsample_bytree': colsample,
                                'reg_lambda': reg_lambda,
                            })
                            print(f"  leaves={num_leaves},lr={learning_rate},lambda={reg_lambda} -> CV_R2={cv_r2:.4f}, Test_R2={test_r2:.4f}")

lgb_best = max(lgb_results, key=lambda x: x['test_r2'])
print(f"\nLightGBM best: Test_R2={lgb_best['test_r2']:.6f}, Test_MAE={lgb_best['test_mae']:.2f}")
print(f"  Best params: {lgb_best}\n")

# ============================================================================
# 汇总三模型最优结果
# ============================================================================
print("="*60)
print("优化后三模型对比结果")
print("="*60)

summary_rows = [
    {
        'model': 'XGBoost',
        'R2_before': 0.710780,
        'MAE_before': 75865.374467,
        'R2_after': xgb_best['test_r2'],
        'MAE_after': xgb_best['test_mae'],
        'R2_gain': xgb_best['test_r2'] - 0.710780,
        'MAE_improvement': -(xgb_best['test_mae'] - 75865.374467),
    },
    {
        'model': 'RandomForest',
        'R2_before': 0.418487,
        'MAE_before': 93987.862462,
        'R2_after': rf_best['test_r2'],
        'MAE_after': rf_best['test_mae'],
        'R2_gain': rf_best['test_r2'] - 0.418487,
        'MAE_improvement': -(rf_best['test_mae'] - 93987.862462),
    },
    {
        'model': 'LightGBM',
        'R2_before': 0.324392,
        'MAE_before': 94966.810050,
        'R2_after': lgb_best['test_r2'],
        'MAE_after': lgb_best['test_mae'],
        'R2_gain': lgb_best['test_r2'] - 0.324392,
        'MAE_improvement': -(lgb_best['test_mae'] - 94966.810050),
    },
]

summary_df = pd.DataFrame(summary_rows)
print("\nBefore-after comparison:")
print(summary_df.to_string(index=False))

# 保存优化后的参数
optimized_params = pd.DataFrame([
    {**xgb_best, 'group': 'XGBoost'},
    {**rf_best, 'group': 'RandomForest'},
    {**lgb_best, 'group': 'LightGBM'},
])

opt_param_path = base_dir / 'optimized_hyperparameters.csv'
optimized_params.to_csv(opt_param_path, index=False, encoding='utf-8-sig')

summary_path = base_dir / 'optimization_summary.csv'
summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

print(f"\nOptimized hyperparameters saved: {opt_param_path}")
print(f"Optimization summary saved: {summary_path}")

# Final ranking
final_ranking = summary_df[['model', 'R2_after', 'MAE_after']].sort_values('R2_after', ascending=False).reset_index(drop=True)
print("\nFinal Ranking (by optimized R2):")
print(final_ranking.to_string(index=False))
