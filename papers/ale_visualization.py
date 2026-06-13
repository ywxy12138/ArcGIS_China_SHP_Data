"""
ALE 分析结果可视化脚本
生成模型对比、特征热力图、曲线对比等多角度图表
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_results() -> tuple:
    base_dir = Path.cwd()
    curves = pd.read_csv(base_dir / "ale_curve_details.csv")
    summary = pd.read_csv(base_dir / "ale_feature_impact_summary.csv")
    model_avg = pd.read_csv(base_dir / "ale_model_average_impact.csv")
    return curves, summary, model_avg, base_dir


def plot_model_average_comparison(model_avg: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制三模型平均影响对比图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    metrics = ["avg_mean_abs_ale", "avg_max_abs_ale", "avg_ale_range"]
    titles = ["平均绝对 ALE", "平均最大绝对 ALE", "平均 ALE 范围"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        data = model_avg.sort_values(metric, ascending=False)
        bars = ax.bar(data["model"], data[metric], color=color, alpha=0.75, edgecolor="black", linewidth=1.5)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("值（人口数）", fontsize=11)
        ax.set_xlabel("模型", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.0f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("三模型 ALE 平均影响对比", fontsize=14, fontweight="bold", y=1.00)
    fig.savefig(output_dir / "ale_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 模型对比图: ale_model_comparison.png")


def plot_feature_importance_by_model(summary: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制各模型 Top10 特征重要性排序
    """
    models = summary["model"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    colors_by_model = {"XGBoost": "#FF6B6B", "RandomForest": "#4ECDC4", "LightGBM": "#45B7D1"}

    for ax, model in zip(axes, sorted(models)):
        data = summary[summary["model"] == model].nlargest(10, "mean_abs_ale")
        ax.barh(
            range(len(data)),
            data["mean_abs_ale"],
            color=colors_by_model.get(model, "#95E1D3"),
            alpha=0.75,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data["feature"], fontsize=9)
        ax.set_xlabel("平均绝对 ALE", fontsize=10)
        ax.set_title(f"{model} Top10 特征", fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.suptitle("各模型的 Top10 特征影响排序", fontsize=13, fontweight="bold")
    fig.savefig(output_dir / "ale_feature_ranking_by_model.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 特征排序图: ale_feature_ranking_by_model.png")


def plot_feature_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制特征×模型热力图，显示平均绝对 ALE
    """
    pivot = summary.pivot(index="feature", columns="model", values="mean_abs_ale")
    pivot = pivot.sort_values(["LightGBM", "XGBoost", "RandomForest"], ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", color="black", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("平均绝对 ALE", fontsize=11)

    ax.set_xlabel("模型", fontsize=12, fontweight="bold")
    ax.set_ylabel("特征", fontsize=12, fontweight="bold")
    fig.suptitle("特征×模型 ALE 值热力图（Top15 特征）", fontsize=13, fontweight="bold")

    fig.savefig(output_dir / "ale_feature_model_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 热力图: ale_feature_model_heatmap.png")


def plot_ale_curve_comparison_by_feature(curves: pd.DataFrame, feature_list: list, output_dir: Path) -> None:
    """
    绘制指定特征在三模型中的 ALE 曲线对比
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()

    colors = {"XGBoost": "#FF6B6B", "RandomForest": "#4ECDC4", "LightGBM": "#45B7D1"}

    for idx, feature in enumerate(feature_list):
        ax = axes[idx]
        for model, color in colors.items():
            curve = curves[(curves["model"] == model) & (curves["feature"] == feature)].sort_values("center")
            if not curve.empty:
                ax.plot(curve["center"], curve["centered_cumulative_ale"], label=model, linewidth=2.5, color=color)

        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(feature, fontsize=10, fontweight="bold")
        ax.set_xlabel("特征值", fontsize=9)
        ax.set_ylabel("中心化累积 ALE", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("关键特征 ALE 曲线跨模型对比", fontsize=13, fontweight="bold")
    fig.savefig(output_dir / "ale_curve_feature_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 曲线对比图: ale_curve_feature_comparison.png")


def plot_ale_statistics_scatter(summary: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制特征 ALE 各维度的散点图（均值 vs 标准差 vs 范围）
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    colors = {"XGBoost": "#FF6B6B", "RandomForest": "#4ECDC4", "LightGBM": "#45B7D1"}
    models = summary["model"].unique()

    # 均值 vs 标准差
    ax = axes[0]
    for model in models:
        data = summary[summary["model"] == model]
        ax.scatter(data["mean_abs_ale"], data["std_ale"], s=100, alpha=0.6, label=model, color=colors[model])
    ax.set_xlabel("平均绝对 ALE", fontsize=11)
    ax.set_ylabel("标准差 ALE", fontsize=11)
    ax.set_title("平均影响 vs 波动性", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 均值 vs 最大值
    ax = axes[1]
    for model in models:
        data = summary[summary["model"] == model]
        ax.scatter(data["mean_abs_ale"], data["max_abs_ale"], s=100, alpha=0.6, label=model, color=colors[model])
    ax.set_xlabel("平均绝对 ALE", fontsize=11)
    ax.set_ylabel("最大绝对 ALE", fontsize=11)
    ax.set_title("平均影响 vs 峰值", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 范围 vs 平均值
    ax = axes[2]
    for model in models:
        data = summary[summary["model"] == model]
        ax.scatter(data["ale_range"], data["mean_abs_ale"], s=100, alpha=0.6, label=model, color=colors[model])
    ax.set_xlabel("ALE 范围", fontsize=11)
    ax.set_ylabel("平均绝对 ALE", fontsize=11)
    ax.set_title("影响范围 vs 平均强度", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("ALE 统计特征分析", fontsize=13, fontweight="bold")
    fig.savefig(output_dir / "ale_statistics_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 统计散点图: ale_statistics_scatter.png")


def plot_model_feature_distribution(summary: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制各模型特征影响分布的箱线图
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    models = sorted(summary["model"].unique())
    data_by_model = [summary[summary["model"] == model]["mean_abs_ale"].values for model in models]

    colors_list = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    bp = ax.boxplot(data_by_model, labels=models, patch_artist=True, widths=0.6)

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("平均绝对 ALE", fontsize=12, fontweight="bold")
    ax.set_xlabel("模型", fontsize=12, fontweight="bold")
    ax.set_title("各模型特征影响分布（箱线图）", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.savefig(output_dir / "ale_model_distribution_boxplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 箱线图: ale_model_distribution_boxplot.png")


def generate_summary_table(summary: pd.DataFrame, output_dir: Path) -> None:
    """
    生成各模型 Top5 特征的文本汇总表
    """
    output_file = output_dir / "ALE_VISUALIZATION_SUMMARY.txt"

    with open(output_file, "w", encoding="utf-8-sig") as f:
        f.write("=" * 100 + "\n")
        f.write("累积局部效应 (ALE) 分析 - 可视化汇总报告\n")
        f.write("=" * 100 + "\n\n")

        models = sorted(summary["model"].unique())
        for model in models:
            data = summary[summary["model"] == model].nlargest(5, "mean_abs_ale")
            f.write(f"\n【{model} - Top 5 特征】\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'排名':<5} {'特征':<35} {'平均ALE':<15} {'标准差':<15} {'最大值':<15} {'范围':<15}\n")
            f.write("-" * 100 + "\n")

            for rank, (_, row) in enumerate(data.iterrows(), 1):
                f.write(
                    f"{rank:<5} {row['feature']:<35} {row['mean_abs_ale']:<15.2f} "
                    f"{row['std_ale']:<15.2f} {row['max_abs_ale']:<15.2f} {row['ale_range']:<15.2f}\n"
                )

    print(f"[OK] 文本汇总: ALE_VISUALIZATION_SUMMARY.txt")


def main():
    curves, summary, model_avg, output_dir = load_results()

    print("\n开始生成 ALE 可视化图表...\n")

    # 模型平均影响对比
    plot_model_average_comparison(model_avg, output_dir)

    # 各模型特征排序
    plot_feature_importance_by_model(summary, output_dir)

    # 热力图
    plot_feature_heatmap(summary, output_dir)

    # 选择 Top6 特征用于曲线对比
    top_features = summary.nlargest(6, "mean_abs_ale")["feature"].unique().tolist()
    plot_ale_curve_comparison_by_feature(curves, top_features, output_dir)

    # ALE 统计分析
    plot_ale_statistics_scatter(summary, output_dir)

    # 模型分布对比
    plot_model_feature_distribution(summary, output_dir)

    # 文本汇总
    generate_summary_table(summary, output_dir)

    print("\n" + "=" * 50)
    print("所有可视化图表已生成完成！")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
