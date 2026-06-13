from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


# =========================
# 文件路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent

FILE_A = BASE_DIR / "2023_data.xlsx"
FILE_B = BASE_DIR / "China_city_data_base_6.0.xlsx"

# 用户描述中出现过一个重复命名文件，这里做候选兼容
MIGRANT_FILE_CANDIDATES = [
	BASE_DIR / "urban_migrant_outflow_2024urban_migrant_outflow_2024.xls",
	BASE_DIR / "urban_migrant_outflow_2024.xls",
]

# 第一阶段输出（仅 2023 经济数据融合）
OUTPUT_STAGE1_XLSX = BASE_DIR / "city_data_merged_2023_ml.xlsx"
# 第二阶段输出（再拼接 2024 总流出人口数）
OUTPUT_FINAL_XLSX = BASE_DIR / "city_data_merged_2023_with_outflow_ml.xlsx"
OUTPUT_FINAL_CSV = BASE_DIR / "city_data_merged_2023_with_outflow_ml.csv"
OUTPUT_FINAL_CLEAN_XLSX = BASE_DIR / "city_data_merged_2023_with_outflow_ml_clean_3sigma.xlsx"
OUTPUT_FINAL_CLEAN_CSV = BASE_DIR / "city_data_merged_2023_with_outflow_ml_clean_3sigma.csv"


# =========================
# 列名配置
# =========================
# 前表（2023_data.xlsx）要求字段：连接键=行政区划代码
KEEP_COLS_A = [
	"行政区划代码",
	"进出口总额亿元",
	"城镇居民人均可支配收入",
	"农村居民人均可支配收入",
]

# 后表（China_city_data_base_6.0.xlsx）要求字段：连接键=城市代码
KEEP_COLS_B = [
	"年份",
	"省份",
	"城市",
	"城市代码",
	"所属地域",
	"胡焕庸线",
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
]

# 连接键定义
JOIN_KEY_A = "行政区划代码"
JOIN_KEY_B = "城市代码"

# 流出人口表连接字段
MIGRANT_JOIN_KEY = "来源城市ID"
MIGRANT_VALUE_COL = "总流出人口数"
MIGRANT_VALUE_COL_RENAMED = "2024年总流出人口数"

TARGET_YEAR = 2023
YEAR_COL = "年份"

# 3σ 清洗所针对的指标列（不包含标识列/类别列）
INDICATOR_COLS_3SIGMA = [
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
	"2024年总流出人口数",
]


def normalize_col_name(col: str) -> str:
	"""标准化列名，提升列名匹配鲁棒性。"""
	if col is None:
		return ""
	text = str(col).strip()
	replace_map = {
		"（": "(",
		"）": ")",
		"【": "",
		"】": "",
		"“": "",
		"”": "",
		'"': "",
		"'": "",
		" ": "",
	}
	for old, new in replace_map.items():
		text = text.replace(old, new)
	return text.lower()


def build_column_map(columns: Iterable[str]) -> Dict[str, str]:
	"""构建 标准化列名 -> 原始列名 映射。"""
	mapping: Dict[str, str] = {}
	for col in columns:
		key = normalize_col_name(col)
		if key and key not in mapping:
			mapping[key] = col
	return mapping


def resolve_required_columns(df: pd.DataFrame, required_cols: List[str], table_name: str) -> Dict[str, str]:
	"""解析并校验目标列是否存在，返回 目标列名 -> 真实列名。"""
	col_map = build_column_map(df.columns)
	resolved: Dict[str, str] = {}
	missing: List[str] = []

	for wanted in required_cols:
		key = normalize_col_name(wanted)
		real = col_map.get(key)
		if real is None:
			missing.append(wanted)
		else:
			resolved[wanted] = real

	if missing:
		preview = ", ".join(map(str, df.columns[:60]))
		raise KeyError(
			f"{table_name} 缺少以下必需列: {missing}\n"
			f"{table_name} 当前可见列(前60个): {preview}"
		)

	return resolved


def clean_code_series(series: pd.Series) -> pd.Series:
	"""清洗城市编码字段，避免整数/浮点/字符串格式差异影响连接。"""
	return (
		series.astype(str)
		.str.strip()
		.str.replace(r"\\.0$", "", regex=True)
		.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
	)


def locate_migrant_file() -> Path:
	"""按候选列表查找迁移数据文件，兼容用户输入中的重复文件名。"""
	for path in MIGRANT_FILE_CANDIDATES:
		if path.exists():
			return path
	candidate_text = "\n".join(str(p) for p in MIGRANT_FILE_CANDIDATES)
	raise FileNotFoundError(f"未找到迁移数据文件，请确认以下任一文件存在:\n{candidate_text}")


def drop_missing_and_outliers_3sigma(df: pd.DataFrame, indicator_cols: List[str]) -> pd.DataFrame:
	"""对指标列执行缺失值剔除 + 3σ 异常值剔除，返回清洗后的样本。"""
	work_df = df.copy()

	# 指标列统一转数值，非数值内容转为缺失，便于后续统一处理
	for col in indicator_cols:
		work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

	# 步骤1：先删除任一指标缺失的样本
	work_df = work_df.dropna(subset=indicator_cols).copy()

	# 步骤2：按每个指标的均值和标准差执行 3σ 区间过滤
	valid_mask = pd.Series(True, index=work_df.index)
	for col in indicator_cols:
		mean_val = work_df[col].mean()
		std_val = work_df[col].std(ddof=0)
		if pd.isna(std_val) or std_val == 0:
			# 常数列或空列（理论上空列已在缺失值剔除后处理），无需按 3σ 过滤
			continue
		lower = mean_val - 3 * std_val
		upper = mean_val + 3 * std_val
		valid_mask &= work_df[col].between(lower, upper, inclusive="both")

	return work_df.loc[valid_mask].copy()


def main() -> None:
	# 1) 读取主数据表
	df_a = pd.read_excel(FILE_A)
	df_b = pd.read_excel(FILE_B)

	# 2) 解析并校验必需列
	resolved_a = resolve_required_columns(df_a, KEEP_COLS_A + [YEAR_COL], "前表(2023_data.xlsx)")
	resolved_b = resolve_required_columns(df_b, KEEP_COLS_B, "后表(China_city_data_base_6.0.xlsx)")

	# 3) 筛选目标列并重命名到标准字段名
	df_a_selected = df_a[[resolved_a[col] for col in KEEP_COLS_A + [YEAR_COL]]].copy()
	df_b_selected = df_b[[resolved_b[col] for col in KEEP_COLS_B]].copy()

	df_a_selected.rename(columns={resolved_a[col]: col for col in KEEP_COLS_A + [YEAR_COL]}, inplace=True)
	df_b_selected.rename(columns={resolved_b[col]: col for col in KEEP_COLS_B}, inplace=True)

	# 4) 仅保留 2023 年数据（前后两表都先按年份筛选）
	year_numeric_a = pd.to_numeric(df_a_selected[YEAR_COL], errors="coerce")
	df_a_2023 = df_a_selected.loc[year_numeric_a == TARGET_YEAR].copy()

	year_numeric = pd.to_numeric(df_b_selected["年份"], errors="coerce")
	df_b_2023 = df_b_selected.loc[year_numeric == TARGET_YEAR].copy()

	# 5) 清洗连接键并执行第一段连接
	df_a_2023[JOIN_KEY_A] = clean_code_series(df_a_2023[JOIN_KEY_A])
	df_b_2023[JOIN_KEY_B] = clean_code_series(df_b_2023[JOIN_KEY_B])

	stage1 = df_b_2023.merge(
		df_a_2023[[JOIN_KEY_A, "进出口总额亿元", "城镇居民人均可支配收入", "农村居民人均可支配收入"]],
		how="left",
		left_on=JOIN_KEY_B,
		right_on=JOIN_KEY_A,
	)

	# 删除因连接产生的右表键冗余列
	if JOIN_KEY_A in stage1.columns:
		stage1.drop(columns=[JOIN_KEY_A], inplace=True)

	# 6) 第一阶段结果导出
	stage1.to_excel(OUTPUT_STAGE1_XLSX, index=False)

	# 7) 读取迁移数据并提取所需列
	migrant_file = locate_migrant_file()
	df_migrant = pd.read_excel(migrant_file)

	resolved_migrant = resolve_required_columns(
		df_migrant,
		[MIGRANT_JOIN_KEY, MIGRANT_VALUE_COL],
		f"迁移表({migrant_file.name})",
	)

	migrant_selected = df_migrant[[resolved_migrant[MIGRANT_JOIN_KEY], resolved_migrant[MIGRANT_VALUE_COL]]].copy()
	migrant_selected.rename(
		columns={
			resolved_migrant[MIGRANT_JOIN_KEY]: MIGRANT_JOIN_KEY,
			resolved_migrant[MIGRANT_VALUE_COL]: MIGRANT_VALUE_COL_RENAMED,
		},
		inplace=True,
	)

	# 8) 按 城市代码 = 来源城市ID 连接 2024 总流出人口数
	stage1[JOIN_KEY_B] = clean_code_series(stage1[JOIN_KEY_B])
	migrant_selected[MIGRANT_JOIN_KEY] = clean_code_series(migrant_selected[MIGRANT_JOIN_KEY])

	final_df = stage1.merge(
		migrant_selected,
		how="left",
		left_on=JOIN_KEY_B,
		right_on=MIGRANT_JOIN_KEY,
	)

	# 删除迁移表连接键冗余列
	if MIGRANT_JOIN_KEY in final_df.columns:
		final_df.drop(columns=[MIGRANT_JOIN_KEY], inplace=True)

	# 9) 最终结果导出（Excel + CSV）
	final_df.to_excel(OUTPUT_FINAL_XLSX, index=False)
	final_df.to_csv(OUTPUT_FINAL_CSV, index=False, encoding="utf-8-sig")

	# 10) 基于最终结果执行 3σ 清洗（剔除含缺失值和异常值的样本）
	clean_df = drop_missing_and_outliers_3sigma(final_df, INDICATOR_COLS_3SIGMA)
	clean_df.to_excel(OUTPUT_FINAL_CLEAN_XLSX, index=False)
	clean_df.to_csv(OUTPUT_FINAL_CLEAN_CSV, index=False, encoding="utf-8-sig")

	print(f"第一阶段输出: {OUTPUT_STAGE1_XLSX}")
	print(f"最终输出(XLSX): {OUTPUT_FINAL_XLSX}")
	print(f"最终输出(CSV): {OUTPUT_FINAL_CSV}")
	print(f"最终记录数: {len(final_df)}")
	print(f"3σ清洗后输出(XLSX): {OUTPUT_FINAL_CLEAN_XLSX}")
	print(f"3σ清洗后输出(CSV): {OUTPUT_FINAL_CLEAN_CSV}")
	print(f"3σ清洗后记录数: {len(clean_df)}")


if __name__ == "__main__":
	main()
