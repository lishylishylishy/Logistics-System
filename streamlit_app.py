import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI


# ============================================================
# 1. 配置区：只在这里配置固定参数；供应商和目标国家都不写死
# ============================================================
RULE_SHEET_ID = st.secrets["RULE_SHEET_ID"]       # 规则库 Spreadsheet：每个供应商一个 worksheet，例如 4PX、YUNTU
DATA_SHEET_ID = st.secrets["DATA_SHEET_ID"]       # 最终数据 Spreadsheet
AI_API_KEY = st.secrets["API_KEY"]                # AI API Key
GCP_JSON = st.secrets["gcp_json"]                  # Google Service Account JSON

AI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
AI_MODEL = "qwen3.7-plus"
PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]

# 通用业务规则：每条 ID + 国家生成 0~3kg、0.25kg 梯度
STANDARD_WEIGHTS = [
    (0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00),
    (1.00, 1.25), (1.25, 1.50), (1.50, 1.75), (1.75, 2.00),
    (2.00, 2.25), (2.25, 2.50), (2.50, 2.75), (2.75, 3.00),
]


# ============================================================
# 2. 页面
# ============================================================
st.set_page_config(page_title="物流报价解析系统", page_icon="📦", layout="wide")
st.title("📦 物流报价解析系统")
st.caption("上传报价表 → 自动识别供应商 → 加载该供应商 Mapping → 输入目标国家 → 解析 → 预览 → 更新")


# ============================================================
# 3. 基础工具
# ============================================================
def norm(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value).replace("\u3000", " ")).strip()


def sheet_key(value: str) -> str:
    value = norm(value)
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", value)
    return m.group(1) if m else value


def to_bool(value: Any) -> bool:
    return norm(value).lower() in {"true", "1", "yes", "y", "是", "启用"}


def safe_float(value: Any) -> Optional[float]:
    text = norm(value).replace(",", "").replace("，", "")
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def normalize_for_match(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", norm(value).lower())


# ============================================================
# 4. Google Sheets
# ============================================================
@st.cache_resource
def get_gsheet_client():
    try:
        info = json.loads(GCP_JSON, strict=False)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        raise RuntimeError(f"Google认证失败：{e}") from e


def open_spreadsheet(value: str):
    key = sheet_key(value)
    try:
        return get_gsheet_client().open_by_key(key)
    except PermissionError as e:
        raise RuntimeError(
            f"Google Spreadsheet权限不足：{key}。"
            f"请把 gcp_json 中的 client_email 分享给这个 Spreadsheet。"
        ) from e
    except gspread.exceptions.SpreadsheetNotFound as e:
        raise RuntimeError(
            f"找不到 Google Spreadsheet：{key}。请检查 RULE_SHEET_ID / DATA_SHEET_ID。"
        ) from e
    except Exception as e:
        raise RuntimeError(f"打开 Google Spreadsheet 失败：{key}；{e}") from e


def get_worksheet_names(spreadsheet_id: str) -> List[str]:
    return [ws.title for ws in open_spreadsheet(spreadsheet_id).worksheets()]


def load_mapping(mapping_sheet: str) -> pd.DataFrame:
    sh = open_spreadsheet(RULE_SHEET_ID)
    try:
        ws = sh.worksheet(mapping_sheet)
    except gspread.exceptions.WorksheetNotFound as e:
        raise RuntimeError(f"规则库中不存在供应商 Mapping：{mapping_sheet}") from e

    df = pd.DataFrame(ws.get_all_records())
    required = [
        "字段", "是否AI读取", "提取粒度", "记录唯一键",
        "Sheet定位类型", "Sheet定位值",
        "行定位类型", "行定位值",
        "列定位类型", "列定位值",
        "原始提取类型", "Python解析器", "AI指令", "是否必填",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"供应商 Mapping【{mapping_sheet}】缺少列：{', '.join(missing)}")
    return df


def get_country_worksheet(country: str):
    sh = open_spreadsheet(DATA_SHEET_ID)
    try:
        return sh.worksheet(country)
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=country, rows=2000, cols=30)


# ============================================================
# 5. 自动识别供应商
# 逻辑：不需要 Supplier_Config。
# 直接读取规则库中所有 worksheet 名称，和上传文件名 / Excel Sheet名匹配。
# 以后新增供应商：只需在规则库新增一个供应商名称对应的 worksheet。
# ============================================================
def detect_supplier(file_name: str, all_sheets: Dict[str, pd.DataFrame]) -> Tuple[str, str]:
    mapping_sheet_names = get_worksheet_names(RULE_SHEET_ID)
    mapping_sheet_names = [x for x in mapping_sheet_names if x.strip().lower() != "supplier_config"]

    file_norm = normalize_for_match(file_name)
    excel_norms = [normalize_for_match(x) for x in all_sheets.keys()]
    candidates = []

    for mapping_name in mapping_sheet_names:
        token = normalize_for_match(mapping_name)
        if not token:
            continue

        score = 0
        evidence = []

        # 文件名中直接出现供应商名称/代码：最高优先级
        if token in file_norm:
            score += 100
            evidence.append("文件名")

        # Excel Sheet 名中直接出现供应商名称/代码
        hits = [s for s, sn in zip(all_sheets.keys(), excel_norms) if token in sn]
        if hits:
            score += 80 + min(len(hits), 10)
            evidence.append(f"Sheet:{hits[0]}")

        # 显示名称本身可作为 Supplier Code / Mapping Sheet
        if score:
            candidates.append((score, mapping_name, evidence))

    if not candidates:
        raise RuntimeError(
            "无法自动识别供应商。\n"
            f"规则库当前供应商 Mapping：{', '.join(mapping_sheet_names)}\n"
            f"上传文件：{file_name}\n"
            "请确保文件名或 Excel Sheet 名中包含供应商代码/名称，例如 4PX。"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RuntimeError(
            f"供应商识别冲突：{candidates[0][1]} / {candidates[1][1]}。"
            "请让供应商名称在文件名或Sheet名中更明确。"
        )

    return candidates[0][1], "；".join(candidates[0][2])


# ============================================================
# 6. Excel
# ============================================================
@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)


# ============================================================
# 7. Mapping定位
# ============================================================
def get_rule(rules: pd.DataFrame, field: str) -> Dict[str, Any]:
    hit = rules[rules["字段"].astype(str).str.strip() == field]
    if hit.empty:
        raise RuntimeError(f"Mapping中不存在字段：{field}")
    return hit.iloc[0].to_dict()


def match_rule(text: str, rule_type: str, rule_value: str) -> bool:
    text, rule_value = norm(text), norm(rule_value)
    if rule_type == "exact":
        return text == rule_value
    if rule_type == "contains":
        return rule_value in text
    if rule_type == "regex":
        try:
            return bool(re.search(rule_value, text, re.I))
        except re.error as e:
            raise RuntimeError(f"Mapping正则错误：{rule_value}；{e}") from e
    if rule_type == "none":
        return True
    return False


def locate_sheets(all_sheets: Dict[str, pd.DataFrame], rule: Dict[str, Any]) -> List[str]:
    return [
        name for name in all_sheets
        if match_rule(name, rule["Sheet定位类型"], rule["Sheet定位值"])
    ]


def find_cell(df: pd.DataFrame, locator_type: str, locator_value: str, start_row: int = 0, end_row: Optional[int] = None):
    locator_type, locator_value = norm(locator_type), norm(locator_value)
    end_row = len(df) if end_row is None else min(end_row, len(df))

    for r in range(start_row, end_row):
        for c in range(df.shape[1]):
            text = norm(df.iat[r, c])
            if locator_type in {"exact_header", "exact_text"} and text == locator_value:
                return r, c
            if locator_type in {"contains_header", "contains_text"} and locator_value in text:
                return r, c
    return None


def find_country_rows(df: pd.DataFrame, country_rule: Dict[str, Any], target_country: str) -> Tuple[int, int, List[int]]:
    hit = find_cell(df, country_rule["列定位类型"], country_rule["列定位值"])
    if not hit:
        raise RuntimeError(f"无法定位Destination Country列：{country_rule['列定位类型']} / {country_rule['列定位值']}")

    header_row, country_col = hit
    end_row = find_section_start(df)
    target = norm(target_country)
    rows, current_country = [], ""

    for r in range(header_row + 1, end_row):
        cell = norm(df.iat[r, country_col])
        if cell:
            current_country = cell
        if current_country == target:
            rows.append(r)

    return header_row, country_col, rows


def find_section_start(df: pd.DataFrame) -> int:
    anchors = ["价格使用说明", "计重规则", "申报及税费"]
    for r in range(len(df)):
        text = " ".join(norm(v) for v in df.iloc[r].tolist() if norm(v))
        if any(a in text for a in anchors):
            return r
    return len(df)


def extract_section(df: pd.DataFrame, anchor: str) -> str:
    start = None
    for r in range(len(df)):
        text = " ".join(norm(v) for v in df.iloc[r].tolist() if norm(v))
        if anchor in text:
            start = r
            break
    if start is None:
        return ""

    stops = ["价格使用说明", "计重规则", "申报及税费"]
    lines = []
    for r in range(start, len(df)):
        text = " | ".join(norm(v) for v in df.iloc[r].tolist() if norm(v))
        if not text:
            continue
        if r > start and any(x in text for x in stops if x != anchor):
            break
        lines.append(f"Excel Row {r + 1}: {text}")
    return "\n".join(lines)


def locate_column(df: pd.DataFrame, rule: Dict[str, Any]) -> int:
    hit = find_cell(df, rule["列定位类型"], rule["列定位值"])
    if not hit:
        raise RuntimeError(f"无法定位列：{rule['列定位类型']} / {rule['列定位值']}")
    return hit[1]


# ============================================================
# 8. Python通用字段解析
# ============================================================
def extract_id_from_sheet(sheet_name: str) -> str:
    m = re.search(r"[\(（]([A-Za-z0-9]+)[\)）]", sheet_name)
    if not m:
        raise RuntimeError(f"无法从Sheet名称提取ID：{sheet_name}")
    return m.group(1)


def parse_cargo_category(sheet_name: str) -> Optional[str]:
    if "普货" in sheet_name:
        return "Regular"
    if any(x in sheet_name for x in ["带电", "特货", "敏感"]):
        return "Sensitive"
    return None


def parse_number(value: Any) -> Optional[float]:
    return safe_float(value)


def generate_weight_steps(source_min: float, source_max: float) -> List[Tuple[float, float]]:
    if source_min is None or source_max is None or source_min >= source_max:
        return []

    result = []
    for smin, smax in STANDARD_WEIGHTS:
        if smax <= source_min or smin >= source_max:
            continue

        wmin = max(smin, source_min)
        wmax = min(smax, source_max)

        if source_min >= 1 and wmin == smin:
            wmin = 1.00
        elif source_min > 1 and wmin == source_min:
            wmin = round(source_min + 0.01, 2)

        if source_max <= 1 and wmax == smax:
            wmax = 1.00
        elif source_max < 1 and wmax == source_max:
            wmax = round(source_max - 0.01, 2)

        if wmax > wmin:
            result.append((round(wmin, 2), round(wmax, 2)))

    return result


# ============================================================
# 9. AI
# ============================================================
@st.cache_data(show_spinner=False, max_entries=500)
def ai_json(prompt: str, context: str) -> Dict[str, Any]:
    client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
    response = client.chat.completions.create(
        model=AI_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是严谨的物流报价表数据提取专家。"
                    "只能根据提供的原始内容提取，不得猜测。"
                    "不确定就返回null。必须返回合法JSON。"
                ),
            },
            {"role": "user", "content": f"{prompt}\n\n原始内容：\n{context}"},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def ai_extract_metadata(target_country: str, country_context: str, note_text: str, weight_rule_text: str, tax_text: str, rules: pd.DataFrame) -> Dict[str, Any]:
    fields = ["Cargo forbidden", "Time (workday/nature day)", "Volume Limit (cm)", "Volume to Weight parameter", "Pick&Packing/parcel", "Tax Policy"]
    instructions = []
    for field in fields:
        rule = get_rule(rules, field)
        if to_bool(rule["是否AI读取"]):
            instructions.append(f"{field}: {norm(rule['AI指令'])}")

    prompt = f"""
目标国家：{target_country}

只提取目标国家，不要使用其他国家信息。

字段要求：
{chr(10).join(instructions)}

严格返回以下JSON结构：
{{
  "Cargo forbidden": [],
  "Time": {{"min": null, "max": null, "unit": null}},
  "Volume Limit": {{"length_cm": null, "width_cm": null, "height_cm": null, "max_length_cm": null, "max_volume_m3": null, "formula": null, "raw": null}},
  "Volume to Weight parameter": null,
  "Pick&Packing/parcel": null,
  "Tax Policy": {{"delivery_term": null, "fob_limit_usd": null, "cif_limit_usd": null, "raw": null}}
}}

没有明确值返回null；不要猜。
"""

    context = (
        f"目标国家价格/服务行：\n{country_context}\n\n"
        f"价格使用说明：\n{note_text}\n\n"
        f"计重规则：\n{weight_rule_text}\n\n"
        f"申报及税费：\n{tax_text}"
    )
    return ai_json(prompt, context)


def ai_map_weight_rows(target_country: str, weight_rows: List[Dict[str, Any]], rule: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
目标国家：{target_country}

请根据下面真实存在的Excel重量价格行，判断：
1. 源数据最小计费重量 source_min_kg
2. 源数据最大计费重量 source_max_kg
3. 每个标准重量段的 target_max_kg 应取哪一个真实 Excel Row

{norm(rule["AI指令"])}

标准target_max_kg只能是：
0.25、0.50、0.75、1.00、1.25、1.50、1.75、2.00、2.25、2.50、2.75、3.00

严格返回：
{{
  "source_min_kg": null,
  "source_max_kg": null,
  "mapping": [
    {{"target_max_kg": 0.25, "source_excel_row": 12}}
  ]
}}

要求：
- source_excel_row必须是输入中真实存在的Excel Row。
- target_max_kg必须选择能够覆盖该重量的源重量区间。
- 超过源最大计费重量的target_max_kg返回null。
- 不得创建、不存在的行号不得使用。
"""

    return ai_json(prompt, json.dumps(weight_rows, ensure_ascii=False, indent=2))


# ============================================================
# 10. AI结果格式化
# ============================================================
def format_time(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, dict):
        mn, mx, unit = value.get("min"), value.get("max"), value.get("unit")
        if mn is None:
            return None
        return f"{mn} {unit}" if mx is None or mn == mx else f"{mn}~{mx} {unit}"
    return norm(value) or None


def format_dimension(value: Any) -> Optional[str]:
    if not value:
        return None
    if not isinstance(value, dict):
        return norm(value) or None

    parts = []
    if all(value.get(k) is not None for k in ["length_cm", "width_cm", "height_cm"]):
        parts.append(f"{value['length_cm']}×{value['width_cm']}×{value['height_cm']} cm")
    if value.get("max_length_cm") is not None:
        parts.append(f"max_length={value['max_length_cm']}cm")
    if value.get("max_volume_m3") is not None:
        parts.append(f"max_volume={value['max_volume_m3']}m³")
    if value.get("formula"):
        parts.append(f"formula={value['formula']}")
    return "; ".join(parts) or norm(value.get("raw")) or None


def format_tax(value: Any) -> Optional[str]:
    if not value:
        return None
    if not isinstance(value, dict):
        return norm(value) or None

    parts = []
    if value.get("delivery_term"):
        parts.append(str(value["delivery_term"]))
    if value.get("fob_limit_usd") is not None:
        parts.append(f"FOB < {value['fob_limit_usd']} USD")
    if value.get("cif_limit_usd") is not None:
        parts.append(f"CIF < {value['cif_limit_usd']} USD")
    return ", ".join(parts) or norm(value.get("raw")) or None


def format_forbidden(value: Any) -> Optional[str]:
    if isinstance(value, list):
        text = ", ".join(norm(v) for v in value if norm(v))
        return text or None
    return norm(value) or None


# ============================================================
# 11. 解析一条线路Sheet
# ============================================================
def parse_one_sheet(df: pd.DataFrame, sheet_name: str, target_country: str, rules: pd.DataFrame):
    rows, errors = [], []

    try:
        country_rule = get_rule(rules, "Destination Country")
        weight_rule = get_rule(rules, "Weight Range (max kg)")
        freight_rule = get_rule(rules, "RMB /kg")
        parcel_rule = get_rule(rules, "RMB /parcel")

        channel_id = extract_id_from_sheet(sheet_name)
        cargo = parse_cargo_category(sheet_name)

        _, _, country_rows = find_country_rows(df, country_rule, target_country)
        if not country_rows:
            return rows, errors

        weight_col_idx = locate_column(df, weight_rule)
        freight_col_idx = locate_column(df, freight_rule)
        parcel_col_idx = locate_column(df, parcel_rule)

        # 目标国家所有价格行和上下文
        country_context = []
        weight_source_rows = []

        for r in country_rows:
            values = {f"Column_{c+1}": norm(df.iat[r, c]) for c in range(df.shape[1]) if norm(df.iat[r, c])}
            country_context.append({"Excel Row": r + 1, "Values": values})

            raw_weight = norm(df.iat[r, weight_col_idx])
            if raw_weight:
                weight_source_rows.append({
                    "source_excel_row": r + 1,
                    "weight_range_raw": raw_weight,
                    "freight_raw": norm(df.iat[r, freight_col_idx]),
                    "parcel_raw": norm(df.iat[r, parcel_col_idx]),
                })

        if not weight_source_rows:
            return rows, [{"Sheet": sheet_name, "Field": "Weight Range", "Error": "目标国家没有找到重量价格行"}]

        # AI一次处理政策/税务/时效/尺寸/材积系数/P&P
        metadata = ai_extract_metadata(
            target_country,
            json.dumps(country_context, ensure_ascii=False, indent=2),
            extract_section(df, "价格使用说明"),
            extract_section(df, "计重规则"),
            extract_section(df, "申报及税费"),
            rules,
        )

        # AI判断“标准重量max”对应原Excel哪一行
        weight_ai = ai_map_weight_rows(
            target_country,
            weight_source_rows,
            weight_rule,
        )

        source_min = safe_float(weight_ai.get("source_min_kg"))
        source_max = safe_float(weight_ai.get("source_max_kg"))

        if source_min is None or source_max is None:
            return rows, [{"Sheet": sheet_name, "Field": "Weight Range", "Error": "AI无法确定源数据最小/最大计费重量"}]

        weight_steps = generate_weight_steps(source_min, source_max)
        if not weight_steps:
            return rows, [{"Sheet": sheet_name, "Field": "Weight Range", "Error": "无法生成标准重量段"}]

        source_mapping = {}
        for item in weight_ai.get("mapping", []):
            mx = safe_float(item.get("target_max_kg"))
            src = item.get("source_excel_row")
            if mx is None or src is None:
                continue
            try:
                source_mapping[round(mx, 2)] = int(src)
            except (TypeError, ValueError):
                continue

        pick_pack = safe_float(metadata.get("Pick&Packing/parcel"))
        volume_parameter = safe_float(metadata.get("Volume to Weight parameter"))

        target_row_set = {r + 1 for r in country_rows}

        for wmin, wmax in weight_steps:
            source_row = source_mapping.get(round(wmax, 2))

            if source_row is None:
                errors.append({"Sheet": sheet_name, "Field": "Weight Range", "Weight max": wmax, "Error": "AI没有找到对应源价格行"})
                continue

            if source_row not in target_row_set:
                errors.append({"Sheet": sheet_name, "Field": "Weight Range", "Weight max": wmax, "Error": f"AI指定Row {source_row}不属于目标国家"})
                continue

            source_idx = source_row - 1
            rkg = parse_number(df.iat[source_idx, freight_col_idx])
            rparcel = parse_number(df.iat[source_idx, parcel_col_idx])

            # Cell格式异常时，只让AI解析该Cell的数字，不让AI决定线路或价格逻辑
            if rkg is None:
                r = ai_json('从这个Cell中提取RMB/kg数字，只返回{"value":null}，无法明确读取就返回null。', norm(df.iat[source_idx, freight_col_idx]))
                rkg = safe_float(r.get("value"))

            if rparcel is None:
                r = ai_json('从这个Cell中提取RMB/parcel数字，只返回{"value":null}，无法明确读取就返回null。', norm(df.iat[source_idx, parcel_col_idx]))
                rparcel = safe_float(r.get("value"))

            total = None if any(v is None for v in [rkg, rparcel, pick_pack]) else round(wmax * rkg + rparcel + pick_pack, 2)

            rows.append({
                "ID": channel_id,
                "Destination Country": target_country,
                "Cargo Category": cargo,
                "Cargo forbidden": format_forbidden(metadata.get("Cargo forbidden")),
                "Time (workday/nature day)": format_time(metadata.get("Time")),
                "Volume Limit (cm)": format_dimension(metadata.get("Volume Limit")),
                "Volume to Weight parameter": volume_parameter,
                "Weight Range (min kg)": wmin,
                "Weight Range (max kg)": wmax,
                "RMB /kg": rkg,
                "RMB /parcel": rparcel,
                "Pick&Packing/parcel": pick_pack,
                "RMB in total": total,
                "Tax Policy": format_tax(metadata.get("Tax Policy")),
            })

    except Exception as e:
        errors.append({"Sheet": sheet_name, "Field": "Parser", "Error": str(e)})

    return rows, errors


# ============================================================
# 12. 整个Excel解析：只解析用户输入的国家
# ============================================================
def parse_workbook(all_sheets: Dict[str, pd.DataFrame], target_country: str, rules: pd.DataFrame):
    id_rule = get_rule(rules, "ID")
    target_sheets = locate_sheets(all_sheets, id_rule)

    if not target_sheets:
        raise RuntimeError("没有找到符合供应商Mapping的线路Sheet。请检查4PX Mapping中的Sheet定位规则。")

    all_rows, all_errors = [], []
    progress = st.progress(0)
    status = st.empty()

    for i, sheet_name in enumerate(target_sheets, 1):
        status.markdown(f"**解析 [{i}/{len(target_sheets)}]** `{sheet_name}` → `{target_country}`")
        rows, errors = parse_one_sheet(all_sheets[sheet_name], sheet_name, target_country, rules)
        all_rows.extend(rows)
        all_errors.extend(errors)
        progress.progress(i / len(target_sheets))

    progress.empty()
    status.success("✅ 解析完成")

    result = pd.DataFrame(all_rows)
    errors = pd.DataFrame(all_errors)

    if not result.empty:
        result = result.drop_duplicates(subset=PRIMARY_KEYS, keep="last").reset_index(drop=True)

    return result, errors


# ============================================================
# 13. 新旧数据比较
# ============================================================
def compare_data(new_df: pd.DataFrame, old_df: pd.DataFrame):
    if old_df.empty:
        return {
            "new": new_df.copy(),
            "updated": pd.DataFrame(),
            "unchanged": pd.DataFrame(),
            "final": new_df.copy(),
        }

    new, old = new_df.copy(), old_df.copy()

    for c in PRIMARY_KEYS:
        if c not in new.columns:
            new[c] = ""
        if c not in old.columns:
            old[c] = ""

    new["_pk"] = new[PRIMARY_KEYS].astype(str).agg("|".join, axis=1)
    old["_pk"] = old[PRIMARY_KEYS].astype(str).agg("|".join, axis=1)
    old_map = old.set_index("_pk")

    new_rows, updated_rows, unchanged_rows = [], [], []

    for _, n in new.iterrows():
        pk = n["_pk"]

        if pk not in old_map.index:
            new_rows.append(n.drop("_pk").to_dict())
            continue

        o = old_map.loc[pk]
        compare_cols = [c for c in new.columns if c != "_pk" and c in old.columns]
        changed = any(norm(n[c]) != norm(o[c]) for c in compare_cols)

        if changed:
            updated_rows.append(n.drop("_pk").to_dict())
        else:
            unchanged_rows.append(n.drop("_pk").to_dict())

    # 新文件没出现的旧记录暂时保留，避免部分报价表误删旧数据
    untouched = old[~old["_pk"].isin(new["_pk"])].drop(columns="_pk", errors="ignore")

    final = pd.concat(
        [untouched, pd.DataFrame(new_rows), pd.DataFrame(updated_rows), pd.DataFrame(unchanged_rows)],
        ignore_index=True,
    )

    return {
        "new": pd.DataFrame(new_rows),
        "updated": pd.DataFrame(updated_rows),
        "unchanged": pd.DataFrame(unchanged_rows),
        "final": final,
    }


def write_data(ws, df: pd.DataFrame):
    clean = df.fillna("")
    ws.clear()
    ws.update(
        [clean.columns.tolist()] + clean.astype(str).values.tolist(),
        range_name="A1",
    )
    return len(clean)


# ============================================================
# 14. Session State：保证预览后点击“更新”不会丢数据
# ============================================================
def clear_result_state():
    for key in [
        "parsed_df", "errors_df", "comparison",
        "worksheet", "supplier_code", "supplier_name",
        "mapping_sheet", "target_country",
    ]:
        st.session_state.pop(key, None)


# ============================================================
# 15. App界面
# ============================================================
st.subheader("① 上传报价表")
uploaded_file = st.file_uploader("把供应商报价 Excel 拖到这里", type=["xlsx", "xls"])

st.subheader("② 输入目标国家/地区")
target_country = st.text_input(
    "目标国家/地区",
    placeholder="例如：墨西哥、美国、加拿大",
).strip()

if uploaded_file:
    st.info(f"已选择：{uploaded_file.name}（{uploaded_file.size / 1024 / 1024:.1f} MB）")

run = st.button(
    "🚀 识别供应商并开始解析",
    type="primary",
    use_container_width=True,
    disabled=not uploaded_file or not target_country,
)

if run:
    clear_result_state()

    try:
        with st.spinner("正在读取Excel..."):
            all_sheets = load_excel(uploaded_file.getvalue())

        supplier_code, detect_evidence = detect_supplier(uploaded_file.name, all_sheets)
        st.success(f"✅ 自动识别供应商：{supplier_code}")
        st.caption(f"识别依据：{detect_evidence}")

        with st.spinner(f"正在读取 {supplier_code} Mapping..."):
            rules = load_mapping(supplier_code)

        st.success(f"✅ 已加载供应商规则：{supplier_code}")

        with st.spinner(f"正在解析目标国家：{target_country}"):
            parsed_df, errors_df = parse_workbook(all_sheets, target_country, rules)

        if parsed_df.empty:
            st.error(f"❌ 没有提取到【{target_country}】的数据。")
            if not errors_df.empty:
                st.dataframe(errors_df, use_container_width=True)
        else:
            ws = get_country_worksheet(target_country)
            old_df = pd.DataFrame(ws.get_all_records())
            comparison = compare_data(parsed_df, old_df)

            st.session_state["parsed_df"] = parsed_df
            st.session_state["errors_df"] = errors_df
            st.session_state["comparison"] = comparison
            st.session_state["worksheet"] = ws
            st.session_state["supplier_code"] = supplier_code
            st.session_state["supplier_name"] = supplier_code
            st.session_state["mapping_sheet"] = supplier_code
            st.session_state["target_country"] = target_country

    except Exception as e:
        st.error(f"❌ 运行失败：{e}")
        st.exception(e)


# ============================================================
# 16. 解析结果 / 预览 / 更新
# ============================================================
if "parsed_df" in st.session_state:
    parsed_df = st.session_state["parsed_df"]
    errors_df = st.session_state["errors_df"]
    comparison = st.session_state["comparison"]
    ws = st.session_state["worksheet"]
    supplier_code = st.session_state["supplier_code"]
    target_country = st.session_state["target_country"]

    st.divider()
    st.subheader("③ 解析结果")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("解析记录", len(parsed_df))
    c2.metric("新增", len(comparison["new"]))
    c3.metric("更新", len(comparison["updated"]))
    c4.metric("异常", len(errors_df))

    tab1, tab2, tab3, tab4 = st.tabs(["全部结果", "新增", "更新", "异常"])

    with tab1:
        st.dataframe(parsed_df, use_container_width=True, height=600)

    with tab2:
        st.dataframe(comparison["new"], use_container_width=True)

    with tab3:
        st.dataframe(comparison["updated"], use_container_width=True)

    with tab4:
        if errors_df.empty:
            st.success("✅ 没有异常")
        else:
            st.dataframe(errors_df, use_container_width=True)

    st.download_button(
        "⬇️ 下载解析结果 CSV",
        parsed_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{supplier_code}_{target_country}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("④ 更新现有数据")
    st.warning(
        f"唯一键：ID + Destination Country + Weight Range (max kg)。"
        f"同Key的新记录会替换旧记录；本次文件没有出现的旧记录暂时保留。"
    )

    confirm = st.checkbox("我已检查预览结果，确认写入Google Sheet。")

    if st.button(
        "✅ 确认并更新 Google Sheet",
        type="primary",
        use_container_width=True,
        disabled=not confirm,
    ):
        try:
            count = write_data(ws, comparison["final"])
            st.success(f"🎉 更新完成：{target_country} 当前共 {count} 条记录。")
        except Exception as e:
            st.error(f"❌ Google Sheet写入失败：{e}")
            st.exception(e)
