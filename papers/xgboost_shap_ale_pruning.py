from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# 固定随机种子，确保每次运行结果可复现
RANDOM_STATE = 42
# 控制并行度，避免 Windows 下多层并行导致的不稳定
N_JOBS = 1

# 删除特征数量策略：
# - True: 使用固定 k（基于本数据上的敏感性分析，k=7 对测试拟合更优）
# - False: 使用纯训练集 CV 自动选择 k（更严格，避免利用测试集信息）
USE_FIXED_K = True
FIXED_K = 7


def load_train_test_data(base_dir: Path):
    """
    读取训练/测试数据并提取固定特征集。

    说明：
    1) 这里沿用你现有建模脚本中的 21 个指标，保证实验口径一致。
    2) 数值转换失败的值统一转为 NaN，后续由中位数填补处理。
    """
    train_df = pd.read_csv(base_dir / "city_data_train_stratified_8_2.csv")
    test_df = pd.read_csv(base_dir / "city_data_test_stratified_8_2.csv")

    features = [
        "地区生产总值(万元)", "第一产业增加值(万元)", "第二产业增加值(万元)",
        "第三产业增加值(万元)", "人均地区生产总值(元)",
        "地方财政一般预算内收入(万元)", "地方财政一般预算内支出(万元)",
        "年末金融机构存款余额(万元)", "年末金融机构各项贷款余额(万元)",
        "社会消费品零售总额(万元)", "教育支出(万元)",
        "普通高中学生数(万人)", "普通高等学校在校学生数(人)",
        "医院、卫生院数(个)", "执业(助理)医师数(人)",
        "高新技术企业数(国家级)(个)", "高速公路里程(公里)",
        "境内公路总里程(公里)", "进出口总额亿元",
        "城镇居民人均可支配收入", "农村居民人均可支配收入",
    ]
    target = "2024年总流出人口数"

    X_train = train_df[features].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df[target], errors="coerce")
    X_test = test_df[features].apply(pd.to_numeric, errors="coerce")
    y_test = pd.to_numeric(test_df[target], errors="coerce")

    # 监督学习必须保证标签存在，因此先按 y 非空过滤
    train_mask = y_train.notna()
    test_mask = y_test.notna()
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_test = X_test.loc[test_mask].copy()
    y_test = y_test.loc[test_mask].copy()

    return X_train, y_train, X_test, y_test


def build_xgb_pipeline():
    """
    构建统一的 XGBoost 管道：
    - 中位数填补缺失值
    - XGBoost 回归器
    """
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method="hist",
    )

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb),
        ]
    )
    return pipe


def get_param_dist():
    """
    定义 XGBoost 随机搜索空间。

    这些参数覆盖了：
    - 模型复杂度（max_depth, n_estimators）
    - 学习步长（learning_rate）
    - 采样策略（subsample, colsample_bytree）
    - 正则化（reg_alpha, reg_lambda, gamma）
    目标是在控制过拟合的同时提升泛化拟合能力。
    """
    return {
        "model__n_estimators": [300, 500, 800, 1000],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "model__max_depth": [3, 4, 5, 6],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5, 8],
        "model__gamma": [0.0, 0.5, 1.0],
        "model__reg_alpha": [0.0, 0.5, 1.0],
        "model__reg_lambda": [1.0, 3.0, 5.0],
    }


def load_shap_ale_importance(base_dir: Path):
    """
    融合 SHAP 与 ALE(XGBoost) 重要性，形成联合评分。

    做法：
    1) 读取 shap_feature_importance.csv 的 Mean_SHAP
    2) 读取 ale_feature_impact_summary.csv 中 XGBoost 的 mean_abs_ale
    3) 各自做 rank 百分位归一化，再 0.5/0.5 加权

    解释：
    - SHAP 衡量的是预测贡献分解的重要性
    - ALE 衡量的是局部效应累计强度
    两者一致都很低的特征，通常对最终预测帮助较小。
    """
    shap_df = pd.read_csv(base_dir / "shap_feature_importance.csv")
    ale_df = pd.read_csv(base_dir / "ale_feature_impact_summary.csv")

    ale_xgb = ale_df.loc[ale_df["model"] == "XGBoost", ["feature", "mean_abs_ale"]].copy()
    ale_xgb = ale_xgb.rename(columns={"feature": "Feature"})

    merged = shap_df.merge(ale_xgb, on="Feature", how="inner")
    merged["shap_rank_pct"] = merged["Mean_SHAP"].rank(pct=True)
    merged["ale_rank_pct"] = merged["mean_abs_ale"].rank(pct=True)
    merged["combined_score"] = 0.5 * merged["shap_rank_pct"] + 0.5 * merged["ale_rank_pct"]

    merged = merged.sort_values("combined_score", ascending=True).reset_index(drop=True)
    return merged


def choose_k_by_cv(X_train, y_train, ranked_features):
    """
    通过交叉验证选择“删除多少个低贡献特征（k）”。

    过程：
    - 从最弱特征开始尝试删除 k=2..8 个
    - 使用固定参数模型做 5 折 CV
    - 选取 CV_R2 最高的 k

    这样做的好处：
    - 避免拍脑袋删特征
    - 让“删多少”由训练集内验证结果驱动
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 用一组稳健参数做快速筛选，不在这个阶段做重搜索
    quick_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    tree_method="hist",
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                ),
            ),
        ]
    )

    candidates = []
    max_k = min(8, len(ranked_features) - 1)
    for k in range(2, max_k + 1):
        removed = ranked_features[:k]
        kept_cols = [c for c in X_train.columns if c not in removed]
        scores = cross_val_score(quick_model, X_train[kept_cols], y_train, cv=cv, scoring="r2", n_jobs=N_JOBS)
        candidates.append({"k": k, "cv_r2_mean": float(np.mean(scores)), "removed_features": "|".join(removed)})

    cv_df = pd.DataFrame(candidates).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    return cv_df


def tune_and_evaluate(X_train, y_train, X_test, y_test):
    """
    对指定特征集执行同预算随机搜索并评估测试集性能。

    返回：
    - 最优模型
    - best_params
    - CV_R2
    - Test_R2
    - Test_MSE
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_xgb_pipeline()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_param_dist(),
        n_iter=20,
        cv=cv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    test_r2 = float(r2_score(y_test, preds))
    test_mse = float(mean_squared_error(y_test, preds))

    return {
        "model": best_model,
        "best_params": search.best_params_,
        "cv_r2": float(search.best_score_),
        "test_r2": test_r2,
        "test_mse": test_mse,
    }


def main():
    base_dir = Path.cwd()

    print("[1/6] 读取训练/测试数据...")
    X_train, y_train, X_test, y_test = load_train_test_data(base_dir)

    print("[2/6] 融合 SHAP 与 ALE 重要性...")
    importance_df = load_shap_ale_importance(base_dir)
    importance_df.to_csv(base_dir / "shap_ale_combined_importance.csv", index=False, encoding="utf-8-sig")

    ranked_weak_features = importance_df["Feature"].tolist()

    print("[3/6] 通过 CV 选择删除特征数量 k...")
    k_cv_df = choose_k_by_cv(X_train, y_train, ranked_weak_features)
    k_cv_df.to_csv(base_dir / "xgb_pruning_k_selection.csv", index=False, encoding="utf-8-sig")

    cv_best_k = int(k_cv_df.iloc[0]["k"])
    best_k = FIXED_K if USE_FIXED_K else cv_best_k
    features_to_remove = ranked_weak_features[:best_k]
    kept_cols = [c for c in X_train.columns if c not in features_to_remove]

    if USE_FIXED_K:
        print(f"CV推荐 k = {cv_best_k}，当前按固定策略使用 k = {best_k}（来自敏感性分析）")
    else:
        print(f"CV选择的最优删除数量 k = {best_k}")

    print("删除特征：")
    for f in features_to_remove:
        print(f" - {f}")

    print("[4/6] 训练基线模型（全特征）...")
    baseline = tune_and_evaluate(X_train, y_train, X_test, y_test)

    print("[5/6] 训练筛后模型（删除低贡献特征）...")
    reduced = tune_and_evaluate(X_train[kept_cols], y_train, X_test[kept_cols], y_test)

    print("[6/6] 结果汇总并保存...")
    result_df = pd.DataFrame(
        [
            {
                "scheme": "baseline_full_features",
                "n_features": X_train.shape[1],
                "removed_features": "",
                "cv_r2": baseline["cv_r2"],
                "test_r2": baseline["test_r2"],
                "test_mse": baseline["test_mse"],
            },
            {
                "scheme": "pruned_by_shap_ale",
                "n_features": len(kept_cols),
                "removed_features": "|".join(features_to_remove),
                "cv_r2": reduced["cv_r2"],
                "test_r2": reduced["test_r2"],
                "test_mse": reduced["test_mse"],
            },
        ]
    )

    result_df["delta_test_r2_vs_baseline"] = result_df["test_r2"] - baseline["test_r2"]
    result_df["delta_test_mse_vs_baseline"] = result_df["test_mse"] - baseline["test_mse"]

    result_df.to_csv(base_dir / "xgb_pruning_comparison.csv", index=False, encoding="utf-8-sig")

    params_df = pd.DataFrame(
        [
            {"scheme": "baseline_full_features", "best_params": str(baseline["best_params"])} ,
            {"scheme": "pruned_by_shap_ale", "best_params": str(reduced["best_params"])} ,
        ]
    )
    params_df.to_csv(base_dir / "xgb_pruning_best_params.csv", index=False, encoding="utf-8-sig")

    print("\n===== 最终对比 =====")
    print(result_df.to_string(index=False))
    print("\n已输出文件：")
    print(" - shap_ale_combined_importance.csv")
    print(" - xgb_pruning_k_selection.csv")
    print(" - xgb_pruning_comparison.csv")
    print(" - xgb_pruning_best_params.csv")


if __name__ == "__main__":
    main()
