from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


"""
ALE 分析主脚本说明：
1. 先用训练集拟合模型（不使用测试集参与训练）。
2. 再使用测试集特征分布作为 ALE 计算样本。
3. 因此，这里的 ALE 是“基于训练好的模型，仅在测试集上计算”。
4. 这样做的目的是在不使用测试标签参与训练的前提下，观察模型在测试样本分布下的平均边际影响。
"""

def load_data(base_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], str]:
    # 读取分层抽样后的训练集/测试集，后续严格保持“训练集拟合、测试集不参与拟合”的原则。
    train_path = base_dir / "city_data_train_stratified_8_2.csv"
    test_path = base_dir / "city_data_test_stratified_8_2.csv"

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

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df[target_col], errors="coerce")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_test = pd.to_numeric(test_df[target_col], errors="coerce")

    train_mask = y_train.notna()
    test_mask = y_test.notna()

    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    X_test = X_test.loc[test_mask].copy()
    y_test = y_test.loc[test_mask].copy()

    return X_train, y_train, X_test, y_test, feature_cols, target_col


def build_models() -> Dict[str, object]:
    # 这里统一放置三种树模型的固定参数，便于后续复现实验和解释 ALE 结果。
    return {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=800,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=700,
            learning_rate=0.1,
            num_leaves=15,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def compute_ale_1d(
    model,
    X: pd.DataFrame,
    feature: str,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, pd.Series]:
    # 1D ALE 的核心思想：
    # 先按特征分位数划分区间，再在每个区间内只改变当前特征，比较区间上下界预测差值，
    # 最后把局部效应累加并中心化。
    x = X[feature].to_numpy()
    feature_idx = X.columns.get_loc(feature)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, quantiles)
    edges = np.unique(edges)

    if edges.shape[0] < 2:
        empty = pd.DataFrame(
            columns=[
                "feature",
                "bin_idx",
                "left",
                "right",
                "center",
                "n_in_bin",
                "local_effect",
                "cumulative_ale",
                "centered_cumulative_ale",
            ]
        )
        return empty, pd.Series(np.zeros_like(x), index=X.index, name="ale")

    bin_ids = np.digitize(x, edges[1:-1], right=False)
    n_actual_bins = edges.shape[0] - 1

    local_effects = np.zeros(n_actual_bins)
    counts = np.zeros(n_actual_bins, dtype=int)

    for k in range(n_actual_bins):
        mask = bin_ids == k
        idx = np.where(mask)[0]
        counts[k] = int(idx.shape[0])
        if idx.shape[0] == 0:
            continue

        lower = edges[k]
        upper = edges[k + 1]

        x_low = X.to_numpy(copy=True)
        x_high = X.to_numpy(copy=True)

        x_low[idx, feature_idx] = lower
        x_high[idx, feature_idx] = upper

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pred_low = model.predict(pd.DataFrame(x_low[idx], columns=X.columns))
            pred_high = model.predict(pd.DataFrame(x_high[idx], columns=X.columns))

        local_effects[k] = float(np.mean(pred_high - pred_low))

    # 局部效应按区间依次累加，得到未中心化的累计局部效应曲线。
    cumulative = np.cumsum(local_effects)

    sample_ale = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        k = bin_ids[i]
        left = edges[k]
        right = edges[k + 1]
        span = right - left
        frac = 0.0 if span == 0 else (x[i] - left) / span
        prev_cum = 0.0 if k == 0 else cumulative[k - 1]
        sample_ale[i] = prev_cum + frac * local_effects[k]

    # 中心化：使 ALE 在样本上的均值为 0，便于跨模型比较和解读正负方向。
    center_const = float(sample_ale.mean())
    centered_sample_ale = sample_ale - center_const

    centers = (edges[:-1] + edges[1:]) / 2.0
    curve_df = pd.DataFrame(
        {
            "feature": feature,
            "bin_idx": np.arange(n_actual_bins),
            "left": edges[:-1],
            "right": edges[1:],
            "center": centers,
            "n_in_bin": counts,
            "local_effect": local_effects,
            "cumulative_ale": cumulative,
            "centered_cumulative_ale": cumulative - center_const,
        }
    )

    sample_ale_series = pd.Series(centered_sample_ale, index=X.index, name="ale")
    return curve_df, sample_ale_series


def plot_top_features(curves: pd.DataFrame, summary: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    model_summary = summary[summary["model"] == model_name].sort_values("mean_abs_ale", ascending=False)
    top_features = model_summary.head(6)["feature"].tolist()

    if not top_features:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, feature in zip(axes, top_features):
        c = curves[(curves["model"] == model_name) & (curves["feature"] == feature)]
        ax.plot(c["center"], c["centered_cumulative_ale"], linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(feature, fontsize=9)
        ax.set_xlabel("Feature value", fontsize=8)
        ax.set_ylabel("Centered ALE", fontsize=8)

    for i in range(len(top_features), len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"{model_name} Top-6 Feature ALE Curves", fontsize=13)
    out_path = output_dir / f"ale_top6_{model_name.lower()}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    base_dir = Path.cwd()
    X_train, y_train, X_test, y_test, feature_cols, _ = load_data(base_dir)

    # 模型训练只使用训练集。
    # 计算 ALE 时，仅使用测试集特征样本分布。
    # 注意：这不使用测试集标签参与训练，只用于解释模型在测试分布下的局部响应。
    X_for_ale = X_test.reset_index(drop=True)

    models = build_models()

    all_curve_rows = []
    summary_rows = []

    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)

        imputer = pipe.named_steps["imputer"]
        fitted_model = pipe.named_steps["model"]

        # 对测试集 ALE 输入做同样的缺失值填补，保证与训练时输入口径一致。
        X_for_ale_imputed = pd.DataFrame(
            imputer.transform(X_for_ale),
            columns=feature_cols,
        )

        for feature in feature_cols:
            # 这里输出的是：当前模型在“测试集分布”上的该特征 ALE 曲线与样本级 ALE 值。
            curve_df, ale_samples = compute_ale_1d(
                model=fitted_model,
                X=X_for_ale_imputed,
                feature=feature,
                n_bins=10,
            )

            if curve_df.empty:
                continue

            curve_df.insert(0, "model", model_name)
            all_curve_rows.append(curve_df)

            summary_rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "mean_abs_ale": float(np.mean(np.abs(ale_samples))),
                    "std_ale": float(np.std(ale_samples)),
                    "max_abs_ale": float(np.max(np.abs(ale_samples))),
                    "ale_range": float(np.max(ale_samples) - np.min(ale_samples)),
                }
            )

    curves_df = pd.concat(all_curve_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(["model", "mean_abs_ale"], ascending=[True, False]).reset_index(drop=True)

    model_avg_impact = (
        summary_df.groupby("model", as_index=False)
        .agg(
            avg_mean_abs_ale=("mean_abs_ale", "mean"),
            avg_max_abs_ale=("max_abs_ale", "mean"),
            avg_ale_range=("ale_range", "mean"),
        )
        .sort_values("avg_mean_abs_ale", ascending=False)
        .reset_index(drop=True)
    )

    curves_path = base_dir / "ale_curve_details.csv"
    summary_path = base_dir / "ale_feature_impact_summary.csv"
    model_impact_path = base_dir / "ale_model_average_impact.csv"

    curves_df.to_csv(curves_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    model_avg_impact.to_csv(model_impact_path, index=False, encoding="utf-8-sig")

    for model_name in models.keys():
        plot_top_features(curves_df, summary_df, model_name, base_dir)

    print("ALE 分析完成")
    print(f"1) 分箱-局部效应-累积-中心化明细: {curves_path}")
    print(f"2) 特征平均影响汇总: {summary_path}")
    print(f"3) 模型平均影响汇总: {model_impact_path}")


if __name__ == "__main__":
    main()
