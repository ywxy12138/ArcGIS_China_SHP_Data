from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def main() -> None:
    base_dir = Path.cwd()
    train_path = base_dir / "city_data_train_stratified_8_2.csv"
    test_path = base_dir / "city_data_test_stratified_8_2.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "未找到分层抽样后的训练/测试集，请先生成 city_data_train_stratified_8_2.csv 和 city_data_test_stratified_8_2.csv"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [
        "地区生产总值(万元)",
        "第一产业增加值(万元)",
        "第二产业增加值(万元)",
        "第三产业增加值(万元)",
        "人均地区生产总值(元)",
        "地方财政一般预算内收入(万元)",
        "地方财政一般预算内支出(万元)",
        "年末金融机构存款余额(万元)",
        "年末金融机构各项贷款余额(万元)",
        "社会消费品零售总额(万元)",
        "教育支出(万元)",
        "普通高中学生数(万人)",
        "普通高等学校在校学生数(人)",
        "医院、卫生院数(个)",
        "执业(助理)医师数(人)",
        "高新技术企业数(国家级)(个)",
        "高速公路里程(公里)",
        "境内公路总里程(公里)",
        "进出口总额亿元",
        "城镇居民人均可支配收入",
        "农村居民人均可支配收入",
    ]
    target_col = "2024年总流出人口数"

    missing_train = [c for c in feature_cols if c not in train_df.columns]
    missing_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_train:
        raise KeyError(f"训练集缺少特征列: {missing_train}")
    if missing_test:
        raise KeyError(f"测试集缺少特征列: {missing_test}")
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"训练集或测试集缺少目标列: {target_col}")

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").copy()
    y_train = pd.to_numeric(train_df[target_col], errors="coerce").copy()
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").copy()
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").copy()

    train_mask = y_train.notna()
    test_mask = y_test.notna()
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_test = X_test.loc[test_mask].copy()
    y_test = y_test.loc[test_mask].copy()

    models = {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=1.0,
            random_state=780,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=780,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=780,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    rows = []
    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        rows.append(
            {
                "model": name,
                "R2": float(r2_score(y_test, pred)),
                "MAE": float(mean_absolute_error(y_test, pred)),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "feature_count": int(len(feature_cols)),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(["R2", "MAE"], ascending=[False, True]).reset_index(drop=True)
    out_path = base_dir / "rf_lgbm_results_summary.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("随机森林与LightGBM评估完成（同分层抽样与同指标口径）")
    print(result_df)
    print(f"结果文件: {out_path}")


if __name__ == "__main__":
    main()
