"""
Multi-Strategy Hyperparameter Optimization Comparison
Testing 4 optimization methods (GridSearch, RandomSearch, Optuna, Halving) 
across 3 models with 5-fold cross-validation evaluation
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, KFold, cross_validate, HalvingRandomSearchCV
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ============================================================================
# Load Data
# ============================================================================
base_dir = Path.cwd()
train_df = pd.read_csv(base_dir / 'city_data_train_stratified_8_2.csv')
test_df = pd.read_csv(base_dir / 'city_data_test_stratified_8_2.csv')

feature_cols = [
    '地区生产总值(万元)',
    '第一产业增加值(万元)',
    '第二产业增加值(万元)',
    '第三产业增加值(万元)',
    '人均地区生产总值(元)',
    '地方财政一般预算内收入(万元)',
    '地方财政一般预算内支出(万元)',
    '年末金融机构存款余额(万元)',
    '年末金融机构各项贷款余额(万元)',
    '社会消费品零售总额(万元)',
    '教育支出(万元)',
    '普通高中学生数(万人)',
    '普通高等学校在校学生数(人)',
    '医院、卫生院数(个)',
    '执业(助理)医师数(人)',
    '高新技术企业数(国家级)(个)',
    '高速公路里程(公里)',
    '境内公路总里程(公里)',
    '进出口总额亿元',
    '城镇居民人均可支配收入',
    '农村居民人均可支配收入',
]
target_col = '2024年总流出人口数'

X_train = train_df[feature_cols].apply(pd.to_numeric, errors='coerce').copy()
y_train = pd.to_numeric(train_df[target_col], errors='coerce').copy()
X_test = test_df[feature_cols].apply(pd.to_numeric, errors='coerce').copy()
y_test = pd.to_numeric(test_df[target_col], errors='coerce').copy()

train_mask = y_train.notna()
test_mask = y_test.notna()
X_train = X_train.loc[train_mask].copy()
y_train = y_train.loc[train_mask].copy()
X_test = X_test.loc[test_mask].copy()
y_test = y_test.loc[test_mask].copy()

cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("=" * 80)
print("HYPERPARAMETER OPTIMIZATION COMPARISON")
print("=" * 80)

# ============================================================================
# 1. XGBOOST Optimization
# ============================================================================
print("\n[1/3] XGBoost Optimization...")

# Strategy 1: GridSearchCV (limited grid for speed)
print("  - GridSearchCV...", end=" ", flush=True)
xgb_grid_params = {
    'model__n_estimators': [300, 400, 500],
    'model__max_depth': [3, 4, 5],
    'model__learning_rate': [0.03, 0.05],
    'model__subsample': [0.7, 0.8],
}
pipe_xgb_grid = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1))
])
grid_xgb = GridSearchCV(pipe_xgb_grid, xgb_grid_params, scoring='r2', cv=cv5, n_jobs=-1, verbose=0)
grid_xgb.fit(X_train, y_train)
pred_xgb_grid = grid_xgb.best_estimator_.predict(X_test)
r2_xgb_grid = r2_score(y_test, pred_xgb_grid)
mae_xgb_grid = mean_absolute_error(y_test, pred_xgb_grid)
results.append({
    'Model': 'XGBoost',
    'Strategy': 'GridSearch',
    'R2': r2_xgb_grid,
    'MAE': mae_xgb_grid,
    'CV_Score': grid_xgb.best_score_,
})
print(f"R2={r2_xgb_grid:.6f}, MAE={mae_xgb_grid:.1f}")

# Strategy 2: RandomizedSearchCV
print("  - RandomizedSearchCV...", end=" ", flush=True)
xgb_rand_params = {
    'model__n_estimators': [200, 300, 400, 500, 600],
    'model__max_depth': [2, 3, 4, 5, 6],
    'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__min_child_weight': [1, 3, 5],
}
rand_xgb = RandomizedSearchCV(pipe_xgb_grid, xgb_rand_params, n_iter=50, scoring='r2', 
                              cv=cv5, n_jobs=-1, random_state=42, verbose=0)
rand_xgb.fit(X_train, y_train)
pred_xgb_rand = rand_xgb.best_estimator_.predict(X_test)
r2_xgb_rand = r2_score(y_test, pred_xgb_rand)
mae_xgb_rand = mean_absolute_error(y_test, pred_xgb_rand)
results.append({
    'Model': 'XGBoost',
    'Strategy': 'RandomSearch',
    'R2': r2_xgb_rand,
    'MAE': mae_xgb_rand,
    'CV_Score': rand_xgb.best_score_,
})
print(f"R2={r2_xgb_rand:.6f}, MAE={mae_xgb_rand:.1f}")

# Strategy 3: Halving Random Search
print("  - HalvingRandomSearch...", end=" ", flush=True)
halving_xgb = HalvingRandomSearchCV(pipe_xgb_grid, xgb_rand_params, n_candidates=40, 
                                    scoring='r2', cv=cv5, n_jobs=-1, random_state=42, verbose=0)
halving_xgb.fit(X_train, y_train)
pred_xgb_halving = halving_xgb.best_estimator_.predict(X_test)
r2_xgb_halving = r2_score(y_test, pred_xgb_halving)
mae_xgb_halving = mean_absolute_error(y_test, pred_xgb_halving)
results.append({
    'Model': 'XGBoost',
    'Strategy': 'HalvingSearch',
    'R2': r2_xgb_halving,
    'MAE': mae_xgb_halving,
    'CV_Score': halving_xgb.best_score_,
})
print(f"R2={r2_xgb_halving:.6f}, MAE={mae_xgb_halving:.1f}")

# Strategy 4: Optuna (if available)
if OPTUNA_AVAILABLE:
    print("  - Optuna (Bayesian)...", end=" ", flush=True)
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        }
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **params))
        ])
        scores = cross_validate(pipe, X_train, y_train, cv=cv5, scoring='r2', n_jobs=-1)
        return float(np.mean(scores['test_score']))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=False)
    
    best_params_xgb_optuna = study_xgb.best_params.copy()
    best_params_xgb_optuna['objective'] = 'reg:squarederror'
    best_params_xgb_optuna['random_state'] = 42
    best_params_xgb_optuna['n_jobs'] = -1
    
    pipe_xgb_optuna = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBRegressor(**best_params_xgb_optuna))
    ])
    pipe_xgb_optuna.fit(X_train, y_train)
    pred_xgb_optuna = pipe_xgb_optuna.predict(X_test)
    r2_xgb_optuna = r2_score(y_test, pred_xgb_optuna)
    mae_xgb_optuna = mean_absolute_error(y_test, pred_xgb_optuna)
    results.append({
        'Model': 'XGBoost',
        'Strategy': 'Optuna',
        'R2': r2_xgb_optuna,
        'MAE': mae_xgb_optuna,
        'CV_Score': study_xgb.best_value,
    })
    print(f"R2={r2_xgb_optuna:.6f}, MAE={mae_xgb_optuna:.1f}")

# ============================================================================
# 2. RANDOMFOREST Optimization
# ============================================================================
print("\n[2/3] RandomForest Optimization...")

# Strategy 1: GridSearchCV
print("  - GridSearchCV...", end=" ", flush=True)
rf_grid_params = {
    'model__n_estimators': [300, 500, 700],
    'model__max_depth': [15, 20, 25],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
}
pipe_rf_grid = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
])
grid_rf = GridSearchCV(pipe_rf_grid, rf_grid_params, scoring='r2', cv=cv5, n_jobs=-1, verbose=0)
grid_rf.fit(X_train, y_train)
pred_rf_grid = grid_rf.best_estimator_.predict(X_test)
r2_rf_grid = r2_score(y_test, pred_rf_grid)
mae_rf_grid = mean_absolute_error(y_test, pred_rf_grid)
results.append({
    'Model': 'RandomForest',
    'Strategy': 'GridSearch',
    'R2': r2_rf_grid,
    'MAE': mae_rf_grid,
    'CV_Score': grid_rf.best_score_,
})
print(f"R2={r2_rf_grid:.6f}, MAE={mae_rf_grid:.1f}")

# Strategy 2: RandomizedSearchCV
print("  - RandomizedSearchCV...", end=" ", flush=True)
rf_rand_params = {
    'model__n_estimators': [200, 300, 400, 500, 600, 800],
    'model__max_depth': [10, 15, 20, 25, 30, None],
    'model__min_samples_split': [2, 3, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None],
}
rand_rf = RandomizedSearchCV(pipe_rf_grid, rf_rand_params, n_iter=50, scoring='r2',
                             cv=cv5, n_jobs=-1, random_state=42, verbose=0)
rand_rf.fit(X_train, y_train)
pred_rf_rand = rand_rf.best_estimator_.predict(X_test)
r2_rf_rand = r2_score(y_test, pred_rf_rand)
mae_rf_rand = mean_absolute_error(y_test, pred_rf_rand)
results.append({
    'Model': 'RandomForest',
    'Strategy': 'RandomSearch',
    'R2': r2_rf_rand,
    'MAE': mae_rf_rand,
    'CV_Score': rand_rf.best_score_,
})
print(f"R2={r2_rf_rand:.6f}, MAE={mae_rf_rand:.1f}")

# Strategy 3: Halving Random Search
print("  - HalvingRandomSearch...", end=" ", flush=True)
halving_rf = HalvingRandomSearchCV(pipe_rf_grid, rf_rand_params, n_candidates=40,
                                   scoring='r2', cv=cv5, n_jobs=-1, random_state=42, verbose=0)
halving_rf.fit(X_train, y_train)
pred_rf_halving = halving_rf.best_estimator_.predict(X_test)
r2_rf_halving = r2_score(y_test, pred_rf_halving)
mae_rf_halving = mean_absolute_error(y_test, pred_rf_halving)
results.append({
    'Model': 'RandomForest',
    'Strategy': 'HalvingSearch',
    'R2': r2_rf_halving,
    'MAE': mae_rf_halving,
    'CV_Score': halving_rf.best_score_,
})
print(f"R2={r2_rf_halving:.6f}, MAE={mae_rf_halving:.1f}")

# Strategy 4: Optuna
if OPTUNA_AVAILABLE:
    print("  - Optuna (Bayesian)...", end=" ", flush=True)
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 10, 35),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(random_state=42, n_jobs=-1, **params))
        ])
        scores = cross_validate(pipe, X_train, y_train, cv=cv5, scoring='r2', n_jobs=-1)
        return float(np.mean(scores['test_score']))
    
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=50, show_progress_bar=False)
    
    best_params_rf_optuna = study_rf.best_params.copy()
    best_params_rf_optuna['random_state'] = 42
    best_params_rf_optuna['n_jobs'] = -1
    
    pipe_rf_optuna = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(**best_params_rf_optuna))
    ])
    pipe_rf_optuna.fit(X_train, y_train)
    pred_rf_optuna = pipe_rf_optuna.predict(X_test)
    r2_rf_optuna = r2_score(y_test, pred_rf_optuna)
    mae_rf_optuna = mean_absolute_error(y_test, pred_rf_optuna)
    results.append({
        'Model': 'RandomForest',
        'Strategy': 'Optuna',
        'R2': r2_rf_optuna,
        'MAE': mae_rf_optuna,
        'CV_Score': study_rf.best_value,
    })
    print(f"R2={r2_rf_optuna:.6f}, MAE={mae_rf_optuna:.1f}")

# ============================================================================
# 3. LIGHTGBM Optimization
# ============================================================================
print("\n[3/3] LightGBM Optimization...")

# Strategy 1: GridSearchCV
print("  - GridSearchCV...", end=" ", flush=True)
lgb_grid_params = {
    'model__n_estimators': [500, 700, 900],
    'model__learning_rate': [0.01, 0.03, 0.05],
    'model__num_leaves': [25, 31, 40],
    'model__subsample': [0.8, 0.9],
}
pipe_lgb_grid = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1))
])
grid_lgb = GridSearchCV(pipe_lgb_grid, lgb_grid_params, scoring='r2', cv=cv5, n_jobs=-1, verbose=0)
grid_lgb.fit(X_train, y_train)
pred_lgb_grid = grid_lgb.best_estimator_.predict(X_test)
r2_lgb_grid = r2_score(y_test, pred_lgb_grid)
mae_lgb_grid = mean_absolute_error(y_test, pred_lgb_grid)
results.append({
    'Model': 'LightGBM',
    'Strategy': 'GridSearch',
    'R2': r2_lgb_grid,
    'MAE': mae_lgb_grid,
    'CV_Score': grid_lgb.best_score_,
})
print(f"R2={r2_lgb_grid:.6f}, MAE={mae_lgb_grid:.1f}")

# Strategy 2: RandomizedSearchCV
print("  - RandomizedSearchCV...", end=" ", flush=True)
lgb_rand_params = {
    'model__n_estimators': [300, 500, 700, 1000],
    'model__learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1],
    'model__num_leaves': [20, 25, 31, 40, 50],
    'model__subsample': [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'model__reg_alpha': [0.0, 0.1, 1.0],
    'model__reg_lambda': [0.1, 1.0, 5.0],
}
rand_lgb = RandomizedSearchCV(pipe_lgb_grid, lgb_rand_params, n_iter=50, scoring='r2',
                              cv=cv5, n_jobs=-1, random_state=42, verbose=0)
rand_lgb.fit(X_train, y_train)
pred_lgb_rand = rand_lgb.best_estimator_.predict(X_test)
r2_lgb_rand = r2_score(y_test, pred_lgb_rand)
mae_lgb_rand = mean_absolute_error(y_test, pred_lgb_rand)
results.append({
    'Model': 'LightGBM',
    'Strategy': 'RandomSearch',
    'R2': r2_lgb_rand,
    'MAE': mae_lgb_rand,
    'CV_Score': rand_lgb.best_score_,
})
print(f"R2={r2_lgb_rand:.6f}, MAE={mae_lgb_rand:.1f}")

# Strategy 3: Halving Random Search
print("  - HalvingRandomSearch...", end=" ", flush=True)
halving_lgb = HalvingRandomSearchCV(pipe_lgb_grid, lgb_rand_params, n_candidates=40,
                                    scoring='r2', cv=cv5, n_jobs=-1, random_state=42, verbose=0)
halving_lgb.fit(X_train, y_train)
pred_lgb_halving = halving_lgb.best_estimator_.predict(X_test)
r2_lgb_halving = r2_score(y_test, pred_lgb_halving)
mae_lgb_halving = mean_absolute_error(y_test, pred_lgb_halving)
results.append({
    'Model': 'LightGBM',
    'Strategy': 'HalvingSearch',
    'R2': r2_lgb_halving,
    'MAE': mae_lgb_halving,
    'CV_Score': halving_lgb.best_score_,
})
print(f"R2={r2_lgb_halving:.6f}, MAE={mae_lgb_halving:.1f}")

# Strategy 4: Optuna
if OPTUNA_AVAILABLE:
    print("  - Optuna (Bayesian)...", end=" ", flush=True)
    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        }
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1, **params))
        ])
        scores = cross_validate(pipe, X_train, y_train, cv=cv5, scoring='r2', n_jobs=-1)
        return float(np.mean(scores['test_score']))
    
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=False)
    
    best_params_lgb_optuna = study_lgb.best_params.copy()
    best_params_lgb_optuna['random_state'] = 42
    best_params_lgb_optuna['n_jobs'] = -1
    best_params_lgb_optuna['verbosity'] = -1
    
    pipe_lgb_optuna = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', LGBMRegressor(**best_params_lgb_optuna))
    ])
    pipe_lgb_optuna.fit(X_train, y_train)
    pred_lgb_optuna = pipe_lgb_optuna.predict(X_test)
    r2_lgb_optuna = r2_score(y_test, pred_lgb_optuna)
    mae_lgb_optuna = mean_absolute_error(y_test, pred_lgb_optuna)
    results.append({
        'Model': 'LightGBM',
        'Strategy': 'Optuna',
        'R2': r2_lgb_optuna,
        'MAE': mae_lgb_optuna,
        'CV_Score': study_lgb.best_value,
    })
    print(f"R2={r2_lgb_optuna:.6f}, MAE={mae_lgb_optuna:.1f}")

# ============================================================================
# Results Summary
# ============================================================================
print("\n" + "=" * 80)
results_df = pd.DataFrame(results).sort_values(['R2', 'MAE'], ascending=[False, True]).reset_index(drop=True)

print("\nALL OPTIMIZATION STRATEGIES COMPARISON:")
print(results_df.to_string(index=False))

# Find best overall
best_row = results_df.iloc[0]
print("\n" + "=" * 80)
print("BEST OPTIMIZATION STRATEGY OVERALL:")
print(f"Model: {best_row['Model']}")
print(f"Strategy: {best_row['Strategy']}")
print(f"Test Set R2: {best_row['R2']:.6f}")
print(f"Test Set MAE: {best_row['MAE']:.1f}")
print(f"5-Fold CV Score (R2): {best_row['CV_Score']:.6f}")
print("=" * 80)

# Save results
results_path = base_dir / 'optimization_strategies_comparison.csv'
results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
print(f"\nResults saved to: {results_path}")
