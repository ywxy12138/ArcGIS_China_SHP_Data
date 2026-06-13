import csv
import math
import os
from typing import Dict, List, Tuple

try:
	import xlwt
except ImportError:  # pragma: no cover - depends on local env
	xlwt = None

try:
	from openpyxl import Workbook
except ImportError:  # pragma: no cover - depends on local env
	Workbook = None


BASE_DIR = os.path.dirname(__file__)
INFLOW_PATH = os.path.join(BASE_DIR, "tmp_urban_migrant_population_stat_year_1.txt")
SOURCE_PATH = os.path.join(
	BASE_DIR, "tmp_urban_migrant_population_src_stat_year_2.txt"
)
INFLOW_OUT_PATH = os.path.join(BASE_DIR, "urban_migrant_inflow_2024.xls")
OUTFLOW_OUT_PATH = os.path.join(BASE_DIR, "urban_migrant_outflow_2024.xls")
TARGET_YEAR = "2024"
MUNICIPALITIES = {"北京市", "天津市", "重庆市", "上海市"}
ROUNDING_MODE = "nearest"  # nearest | floor | ceil
FLOW_OUT_PATH = os.path.join(BASE_DIR, "urban_migrant_flow_2024.csv")
SHARE_SUM_PATH = os.path.join(BASE_DIR, "urban_migrant_share_sum_2024.xlsx")
SOURCE_EXCEL_PATH = os.path.join(
	BASE_DIR, "tmp_urban_migrant_population_src_stat_year_2.xlsx"
)


def parse_number(raw_value: str) -> float:
	value = raw_value.strip()
	if value == "":
		return 0.0
	try:
		return float(value)
	except ValueError:
		return 0.0


def normalize_share(raw_value: str) -> float:
	share = parse_number(raw_value) / 100.0
	# if share > 1.0:
	# 	share = share / 100.0
	return share


def normalize_city_key(
	province_id: str, province_name: str, city_id: str, city_name: str
) -> Tuple[Tuple[str, str], Dict[str, str]]:
	if city_name in MUNICIPALITIES:
		key = (province_id, province_id)
		return key, {
			"province_id": province_id,
			"province_name": province_name,
			"city_id": province_id,
			"city_name": city_name,
		}

	key = (province_id, city_id)
	return key, {
		"province_id": province_id,
		"province_name": province_name,
		"city_id": city_id,
		"city_name": city_name,
	}


def normalize_city_fields(
	province_id: str, province_name: str, city_id: str, city_name: str
) -> Tuple[str, str, str, str]:
	if city_name in MUNICIPALITIES:
		return province_id, province_name, province_id, city_name
	return province_id, province_name, city_id, city_name


def round_value(value: float) -> int:
	if ROUNDING_MODE == "floor":
		return int(math.floor(value))
	if ROUNDING_MODE == "ceil":
		return int(math.ceil(value))
	return int(math.floor(value + 0.5))


def read_city_inflow(path: str) -> Tuple[Dict[Tuple[str, str], float], List[List[str]]]:
	inflow_totals: Dict[Tuple[str, str], float] = {}
	inflow_meta: Dict[Tuple[str, str], Dict[str, str]] = {}

	with open(path, "r", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		for row in reader:
			if not row or len(row) < 6:
				continue

			province_id = row[0].strip()
			province_name = row[1].strip()
			city_id = row[2].strip()
			city_name = row[3].strip()
			inflow_total = parse_number(row[4])
			year = row[5].strip()

			if year != TARGET_YEAR:
				continue

			if inflow_total <= 0:
				continue

			key, meta = normalize_city_key(
				province_id, province_name, city_id, city_name
			)
			inflow_totals[key] = inflow_totals.get(key, 0.0) + inflow_total
			if key not in inflow_meta:
				inflow_meta[key] = meta

	inflow_rows: List[List[str]] = []
	for key, total in inflow_totals.items():
		meta = inflow_meta[key]
		inflow_rows.append(
			[
				meta["province_id"],
				meta["province_name"],
				meta["city_id"],
				meta["city_name"],
				str(round(float(total), 6)),
				TARGET_YEAR,
			]
		)

	return inflow_totals, inflow_rows


def compute_city_outflow(
	path: str, inflow_totals: Dict[Tuple[str, str], float]
) -> List[List[str]]:
	outflow_totals: Dict[Tuple[str, str], Dict[str, object]] = {}

	with open(path, "r", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		for row in reader:
			if not row or len(row) < 12:
				continue

			dest_province_id = row[0].strip()
			dest_city_id = row[2].strip()
			dest_city_name = row[3].strip()
			src_province_id = row[4].strip()
			src_province_name = row[5].strip()
			src_city_id = row[6].strip()
			src_city_name = row[7].strip()
			share_raw = row[10]
			year = row[11].strip()

			if year != TARGET_YEAR:
				continue

			dest_key, _ = normalize_city_key(
				dest_province_id, "", dest_city_id, dest_city_name
			)
			if dest_key not in inflow_totals:
				continue

			share = normalize_share(share_raw)
			if share <= 0:
				continue

			inflow_total = inflow_totals[dest_key]
			outflow_value = inflow_total * share

			src_key, src_meta = normalize_city_key(
				src_province_id, src_province_name, src_city_id, src_city_name
			)
			if src_key not in outflow_totals:
				outflow_totals[src_key] = {
					"province_id": src_meta["province_id"],
					"province_name": src_meta["province_name"],
					"city_id": src_meta["city_id"],
					"city_name": src_meta["city_name"],
					"total": 0.0,
				}

			outflow_totals[src_key]["total"] = (
				float(outflow_totals[src_key]["total"]) + outflow_value
			)

	outflow_rows: List[List[str]] = []
	for payload in outflow_totals.values():
		total_value = round_value(float(payload["total"]))
		if total_value < 10000.0:
			continue
		outflow_rows.append(
			[
				payload["province_id"],
				payload["province_name"],
				payload["city_id"],
				payload["city_name"],
				total_value,
				TARGET_YEAR,
			]
		)

	return outflow_rows


def write_excel(path: str, headers: List[str], rows: List[List[str]]) -> None:
	if xlwt is None:
		raise RuntimeError("Missing dependency: xlwt. Install it to write .xls files.")

	workbook = xlwt.Workbook()
	sheet = workbook.add_sheet("Sheet1")

	# 写入表头
	for col_index, value in enumerate(headers):
		sheet.write(0, col_index, value)

	# 写入数据行
	for row_index, row in enumerate(rows, start=1):
		for col_index, value in enumerate(row):
			sheet.write(row_index, col_index, value)

	workbook.save(path)


def write_flow_csv(path: str, rows: List[List[str]]) -> None:
	headers = [
		"省份ID",
		"省份名称",
		"城市ID",
		"城市名称",
		"来源省份ID",
		"来源省份名称",
		"来源城市ID",
		"来源城市名称",
		"来源区县ID",
		"来源区县名称",
		"迁移人口数",
		"年份",
	]

	with open(path, "w", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(headers)
		writer.writerows(rows)


def write_xlsx(path: str, headers: List[str], rows: List[List[str]]) -> None:
	if Workbook is None:
		raise RuntimeError(
			"Missing dependency: openpyxl. Install it to write .xlsx files."
		)

	workbook = Workbook()
	sheet = workbook.active
	sheet.append(headers)
	for row in rows:
		sheet.append(row)
	workbook.save(path)


def convert_source_txt_to_xlsx(path: str) -> None:
	headers = [
		"省份ID",
		"省份名称",
		"城市ID",
		"城市名称",
		"来源省份ID",
		"来源省份名称",
		"来源城市ID",
		"来源城市名称",
		"来源区县ID",
		"来源区县名称",
		"城市外来用户数占比",
		"年份",
	]

	rows: List[List[str]] = []
	with open(path, "r", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		for row in reader:
			if not row or row[11].strip() != TARGET_YEAR:
				continue
			rows.append(row)

	write_xlsx(SOURCE_EXCEL_PATH, headers, rows)


def compute_share_sums(path: str) -> List[List[str]]:
	share_totals: Dict[Tuple[str, str], float] = {}
	share_meta: Dict[Tuple[str, str], Dict[str, str]] = {}

	with open(path, "r", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		for row in reader:
			if not row or len(row) < 12:
				continue

			year = row[11].strip()
			if year != TARGET_YEAR:
				continue

			province_id = row[0].strip()
			province_name = row[1].strip()
			city_id = row[2].strip()
			city_name = row[3].strip()
			share = normalize_share(row[10])

			if share <= 0:
				continue

			(
				province_id,
				province_name,
				city_id,
				city_name,
			) = normalize_city_fields(
				province_id, province_name, city_id, city_name
			)

			key = (province_id, city_id)
			share_totals[key] = share_totals.get(key, 0.0) + share
			if key not in share_meta:
				share_meta[key] = {
					"province_id": province_id,
					"province_name": province_name,
					"city_id": city_id,
					"city_name": city_name,
				}

	rows: List[List[str]] = []
	for key, total in share_totals.items():
		meta = share_meta[key]
		rows.append(
			[
				meta["province_id"],
				meta["province_name"],
				meta["city_id"],
				meta["city_name"],
				total,
				TARGET_YEAR,
			]
		)

	return rows

province = "省直辖县级行政区划"

auto = "自治区直辖县级行政区划"

def main() -> None:
	# 读取城市外来流入人口数据并筛选 2024 年且有流入人口的记录
	inflow_totals, inflow_rows = read_city_inflow(INFLOW_PATH)

	# 计算 2024 年来源城市的总流出人口
	outflow_rows = compute_city_outflow(SOURCE_PATH, inflow_totals)

	# 生成来源城市到目的城市的迁移人口表（以人数替代占比）
	flow_rows: List[List[str]] = []
	with open(SOURCE_PATH, "r", encoding="utf-8") as handle:
		flow_num: Dict[Tuple[Tuple[str, str], Tuple[str, str]], Dict[str, object]] = {}
		reader = csv.reader(handle)
		for row in reader:
			if not row or len(row) < 12:
				continue

			dest_province_id = row[0].strip()
			dest_province_name = row[1].strip()
			dest_city_id = row[2].strip()
			dest_city_name = row[3].strip()
			src_city_name = row[7].strip()
			share_raw = row[10]
			year = row[11].strip()

			if year != TARGET_YEAR or \
				dest_city_name == province or src_city_name == province \
					or dest_city_name == auto or src_city_name == auto:
				continue

			(
				dest_province_id,
				dest_province_name,
				dest_city_id,
				dest_city_name,
			) = normalize_city_fields(
				dest_province_id,
				dest_province_name,
				dest_city_id,
				dest_city_name,
			)
			dest_key, _ = normalize_city_key(
				dest_province_id, dest_province_name, dest_city_id, dest_city_name
			)
			if dest_key not in inflow_totals:
				continue

			share = normalize_share(share_raw)
			if share <= 0:
				continue

			inflow_total = inflow_totals[dest_key]
			flow_value = round_value(inflow_total * share)
			row[0] = dest_province_id
			row[1] = dest_province_name
			row[2] = dest_city_id
			row[3] = dest_city_name
			(
				src_province_id,
				src_province_name,
				src_city_id,
				src_city_name,
			) = normalize_city_fields(
				row[4].strip(),
				row[5].strip(),
				row[6].strip(),
				row[7].strip(),
			)
			row[4] = src_province_id
			row[5] = src_province_name
			row[6] = src_city_id
			row[7] = src_city_name

			src_key, _ = normalize_city_key(
				src_province_id, src_province_name, 
				src_city_id, src_city_name
			)

			key = (src_key, dest_key)

			if key not in flow_num:
				flow_num[key] = {
					"dest_province_id" : row[0],
					"dest_province_name" : row[1],
					"dest_city_id" : row[2],
					"dest_city_name" : row[3],
					"src_province_id" : row[4],
					"src_province_name" : row[5],
					"src_city_id" : row[6],
					"src_city_name" : row[7],
					"src_cnty_id" : row[8],
					"src_cnty_name" : row[9],
					"total" : 0.0,
					"year" : row[11] 
				}

			flow_num[key]["total"] += flow_value

		
		for payload in flow_num.values():
			if payload["total"] < 10000:
				continue
			flow_rows.append(
				[
					payload["dest_province_id"],
					payload["dest_province_name"],
					payload["dest_city_id"],
					payload["dest_city_name"],
					payload["src_province_id"],
					payload["src_province_name"],
					payload["src_city_id"],
					payload["src_city_name"],
					payload["src_cnty_id"],
					payload["src_cnty_name"],
					payload["total"],
					payload["year"]
				]
			)

	# 将筛选后的流入人口数据保存为 Excel
	write_excel(
		INFLOW_OUT_PATH,
		["省份ID", "省份名称", "城市ID", "城市名称", "城市外来人口数", "年份"],
		inflow_rows,
	)

	# 将来源城市总流出人口数据保存为 Excel
	write_excel(
		OUTFLOW_OUT_PATH,
		["来源省份ID", "来源省份名称", "来源城市ID", "来源城市名称", "总流出人口数", "年份"],
		outflow_rows,
	)

	# 保存迁移人口明细表
	write_flow_csv(FLOW_OUT_PATH, flow_rows)

	# 统计各城市来源占比之和并输出为 xlsx
	share_sum_rows = compute_share_sums(SOURCE_PATH)
	write_xlsx(
		SHARE_SUM_PATH,
		["省份ID", "省份名称", "城市ID", "城市名称", "占比合计", "年份"],
		share_sum_rows,
	)

	# 源数据文本转换为 Excel
	# convert_source_txt_to_xlsx(SOURCE_PATH)


if __name__ == "__main__":
	main()
