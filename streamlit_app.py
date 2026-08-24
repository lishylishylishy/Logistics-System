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
# 1. 配置区：需要配置的内容全部集中在这里
# ============================================================
RULE_SHEET_ID = st.secrets["RULE_SHEET_ID"]      # 供应商映射规则库 Spreadsheet ID或URL
DATA_SHEET_ID = st.secrets["DATA_SHEET_ID"]      # 最终数据 Spreadsheet ID或URL
AI_API_KEY = st.secrets["API_KEY"]               # AI API Key
GCP_JSON = st.secrets["gcp_json"]                 # Google Service Account JSON

AI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
AI_MODEL = "qwen3.7-plus"

# 最终记录唯一键：三者相同就是同一条记录，新数据覆盖旧数据
PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]

# 通用固定业务规则：0~3kg，每0.25kg一个梯度
STANDARD_WEIGHTS = [
    (0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00),
    (1.00, 1.25), (1.25, 1.50), (1.50, 1.75), (1.75, 2.00),
    (2.00, 2.25), (2.25, 2.50), (2.50, 2.75), (2.75, 3.00),
]

SUPPLIER_CONFIG_SHEET = "Supplier_Config"


# ============================================================
# 2. 页面
# ============================================================
st.set_page_config(page_title="物流报价解析系统", page_icon="📦", layout="wide")
st.title("📦 物流报价解析系统")
st.caption("上传报价表 → 自动识别供应商 → 加载对应Mapping → 输入目标国家 → 解析与预览 → 确认更新")


# ============================================================
# 3. 基础工具
# ============================================================
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value).replace("\u3000", " ")).strip()


def spreadsheet_key(value: str) -> str:
    value = normalize_text(value)
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", value)
    return m.group(1) if m else value


def as_bool(value: Any) -> bool:
    return normalize_text(value).lower() in {"true", "1", "yes", "y", "是", "启用"}


def safe_float(value: Any) -> Optional[float]:
    text = normalize_text(value).replace(",", "").replace("，", "")
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


# ============================================================
# 4. Google Sheets连接
# ============================================================
@st.cache_resource
def get_gsheet_client():
    try:
        info = json.loads(GCP_JSON, strict=False)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(credentials)
    except Exception as e:
        raise RuntimeError(f"Google认证失败：{e}")


def open_spreadsheet(spreadsheet_id_or_url: str):
    key = spreadsheet_key(spreadsheet_id_or_url)
    try:
        return get_gsheet_client().open_by_key(key)
    except gspread.exceptions.SpreadsheetNotFound as e:
        raise RuntimeError(
            f"无法打开Google Spreadsheet。\n"
            f"Spreadsheet ID：{key}\n"
            f"请确认RULE_SHEET_ID/DATA_SHEET_ID正确，并确认gcp_json中的client_email已共享该Spreadsheet。"
        ) from e
    except gspread.exceptions.APIError as e:
        raise RuntimeError(f"Google Sheets API错误：{e}") from e
    except PermissionError as e:
        raise RuntimeError(
            f"Google Spreadsheet权限不足。\n"
            f"Spreadsheet ID：{key}\n"
            f"请确认gcp_json中的client_email已被共享为Viewer或Editor。"
        ) from e


@st.cache_data(show_spinner=False)
def load_google_records(spreadsheet_id_or_url: str, worksheet_name: str) -> pd.DataFrame:
    sh = open_spreadsheet(spreadsheet_id_or_url)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound as e:
        raise RuntimeError(f"Spreadsheet中找不到工作表：{worksheet_name}") from e
    records = ws.get_all_records()
    return pd.DataFrame(records)


def get_google_worksheet(spreadsheet_id_or_url: str, worksheet_name: str, create=False):
    sh = open_spreadsheet(spreadsheet_id_or_url)
    try:
        return sh.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        if create:
            return sh.add_worksheet(title=worksheet_name, rows=2000, cols=30)
        raise


# ============================================================
# 5. Supplier_Config：自动识别供应商
# ============================================================
def load_supplier_config() -> pd.DataFrame:
    df = load_google_records(RULE_SHEET_ID, SUPPLIER_CONFIG_SHEET)
    required = ["Supplier Code", "Supplier Name", "Enabled", "Detection Type", "Detection Value", "Mapping Sheet"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Supplier_Config缺少列：{', '.join(missing)}")
    return df


def match_text_rule(text: str, rule_type: str, rule_value: str) -> bool:
    text, rule_value = normalize_text(text), normalize_text(rule_value)
    if rule_type == "exact":
        return text == rule_value
    if rule_type == "contains":
        return rule_value in text
    if rule_type == "regex":
        try:
            return bool(re.search(rule_value, text, re.I))
        except re.error as e:
            raise RuntimeError(f"Supplier_Config正则错误：{rule_value}；{e}") from e
    raise RuntimeError(f"不支持的Detection Type：{rule_type}")


def detect_supplier(all_sheets: Dict[str, pd.DataFrame]) -> Tuple[str, str, str]:
    config = load_supplier_config()
    enabled = config[config["Enabled"].apply(as_bool)]
    matches = []
    for _, row in enabled.iterrows():
        rule_type = normalize_text(row["Detection Type"])
        rule_value = normalize_text(row["Detection Value"])
        hit_sheets = [s for s in all_sheets if match_text_rule(s, rule_type, rule_value)]
        if hit_sheets:
            matches.append((len(hit_sheets), row, hit_sheets))
    if not matches:
        raise RuntimeError("无法识别供应商。请检查Supplier_Config中的Detection Type和Detection Value。")
    matches.sort(key=lambda x: x[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        a, b = matches[0][1], matches[1][1]
        raise RuntimeError(f"供应商识别冲突：{a['Supplier Code']} / {b['Supplier Code']}")
    row = matches[0][1]
    return normalize_text(row["Supplier Code"]), normalize_text(row["Supplier Name"]), normalize_text(row["Mapping Sheet"])


# ============================================================
# 6. 读取供应商Mapping
# ============================================================
def load_mapping(mapping_sheet: str) -> pd.DataFrame:
    df = load_google_records(RULE_SHEET_ID, mapping_sheet)
    required = [
        "字段", "是否AI读取", "提取粒度", "记录唯一键",
        "Sheet定位类型", "Sheet定位值",
        "行定位类型", "行定位值",
        "列定位类型", "列定位值",
        "原始提取类型", "Python解析器", "AI指令", "是否必填",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Mapping【{mapping_sheet}】缺少列：{', '.join(missing)}")
    return df


def get_rule(rules: pd.DataFrame, field: str) -> Dict[str, Any]:
    hit = rules[rules["字段"].astype(str).str.strip() == field]
    if hit.empty:
        raise RuntimeError(f"Mapping中没有字段：{field}")
    return hit.iloc[0].to_dict()


# ============================================================
# 7. Excel读取
# ============================================================
@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)


# ============================================================
# 8. Mapping定位引擎
# ============================================================
def locate_sheets(all_sheets: Dict[str, pd.DataFrame], rule: Dict[str, Any]) -> List[str]:
    typ, val = normalize_text(rule["Sheet定位类型"]), normalize_text(rule["Sheet定位值"])
    return [s for s in all_sheets if match_text_rule(s, typ, val)]


def find_cell(df: pd.DataFrame, locator_type: str, locator_value: str, start_row: int = 0, end_row: Optional[int] = None):
    locator_type, locator_value = normalize_text(locator_type), normalize_text(locator_value)
    end_row = len(df) if end_row is None else min(end_row, len(df))
    for r in range(start_row, end_row):
        for c in range(df.shape[1]):
            text = normalize_text(df.iat[r, c])
            if locator_type == "exact_header" and text == locator_value:
                return r, c
            if locator_type == "contains_header" and locator_value in text:
                return r, c
            if locator_type == "exact_text" and text == locator_value:
                return r, c
            if locator_type == "contains_text" and locator_value in text:
                return r, c
    return None


def find_country_column(df: pd.DataFrame, rule: Dict[str, Any]) -> Tuple[int, int]:
    hit = find_cell(df, normalize_text(rule["列定位类型"]), normalize_text(rule["列定位值"]))
    if not hit:
        raise RuntimeError(f"无法定位国家列：{rule['列定位类型']} / {rule['列定位值']}")
    return hit


def find_section_start(df: pd.DataFrame) -> int:
    anchors = ["价格使用说明", "计重规则", "申报及税费"]
    for r in range(len(df)):
        text = " ".join(normalize_text(v) for v in df.iloc[r].tolist() if normalize_text(v))
        if any(a in text for a in anchors):
            return r
    return len(df)


def find_country_rows(df: pd.DataFrame, country_rule: Dict[str, Any], target_country: str) -> Tuple[int, int, List[int]]:
    header_row, country_col = find_country_column(df, country_rule)
    end_row = find_section_start(df)
    rows, current_country = [], ""
    target_country = normalize_text(target_country)
    for r in range(header_row + 1, end_row):
        value = normalize_text(df.iat[r, country_col])
        if value:
            current_country = value
        if current_country == target_country:
            rows.append(r)
    return header_row, country_col, rows


def extract_section(df: pd.DataFrame, anchor: str) -> str:
    start = None
    for r in range(len(df)):
        text = " ".join(normalize_text(v) for v in df.iloc[r].tolist() if normalize_text(v))
        if anchor in text:
            start = r
            break
    if start is None:
        return ""
    stop_anchors = ["价格使用说明", "计重规则", "申报及税费"]
    lines = []
    for r in range(start, len(df)):
        text = " | ".join(normalize_text(v) for v in df.iloc[r].tolist() if normalize_text(v))
        if not text:
            continue
        if r > start and any(a in text for a in stop_anchors if a != anchor):
            break
        lines.append(f"Excel Row {r + 1}: {text}")
    return "\n".join(lines)


# ============================================================
# 9. Python通用解析
# ============================================================
def extract_id(sheet_name: str) -> str:
    m = re.search(r"[\(（]([A-Za-z0-9]+)[\)）]", sheet_name)
    if not m:
        raise RuntimeError(f"无法从Sheet名称提取ID：{sheet_name}")
    return m.group(1)


def cargo_category(sheet_name: str) -> Optional[str]:
    if "普货" in sheet_name:
        return "Regular"
    if any(x in sheet_name for x in ["带电", "特货", "敏感"]):
        return "Sensitive"
    return None


def parse_weight_range(text: Any) -> Tuple[Optional[float], Optional[float]]:
    s = normalize_text(text).upper().replace("KG", "").replace(" ", "")
    patterns = [
        r"(\d+(?:\.\d+)?)<W[≤<=](\d+(?:\.\d+)?)",
        r"W[≤<=](\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*[-~～]\s*(\d+(?:\.\d+)?)",
    ]
    m = re.search(patterns[0], s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(patterns[1], s)
    if m:
        return 0.0, float(m.group(1))
    m = re.search(patterns[2], s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def generate_weight_steps(source_min: float, source_max: float) -> List[Tuple[float, float]]:
    if source_min is None or source_max is None or source_min >= source_max:
        return []
    result = []
    for smin, smax in STANDARD_WEIGHTS:
        if smax <= source_min or smin >= source_max:
            continue
        wmin, wmax = max(smin, source_min), min(smax, source_max)
        if source_min >= 1 and wmin == smin:
            wmin = 1.0
        elif source_min > 1 and wmin == source_min:
            wmin = round(source_min + 0.01, 2)
        if source_max <= 1 and wmax == smax:
            wmax = 1.0
        elif source_max < 1 and wmax == source_max:
            wmax = round(source_max - 0.01, 2)
        if wmax > wmin:
            result.append((round(wmin, 2), round(wmax, 2)))
    return result


# ============================================================
# 10. AI：复杂字段和重量行映射
# ============================================================
@st.cache_data(show_spinner=False, max_entries=500)
def call_ai_json(prompt: str, context: str) -> Dict[str, Any]:
    client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
    response = client.chat.completions.create(
        model=AI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "你是严谨的物流报价表结构化提取专家。只能依据输入内容，不得猜测；无法确定时返回null；必须返回合法JSON。"},
            {"role": "user", "content": f"{prompt}\n\n原始数据：\n{context}"},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def ai_extract_metadata(target_country: str, country_context: str, note_text: str, weight_text: str, tax_text: str, rules: pd.DataFrame) -> Dict[str, Any]:
    ai_fields = ["Cargo forbidden", "Time (workday/nature day)", "Volume Limit (cm)", "Volume to Weight parameter", "Pick&Packing/parcel", "Tax Policy"]
    instructions = []
    for field in ai_fields:
        rule = get_rule(rules, field)
        if as_bool(rule["是否AI读取"]):
            instructions.append(f"{field}: {normalize_text(rule['AI指令'])}")
    prompt = f"""
目标国家：{target_country}
只提取该国家，不要使用其他国家的信息。

字段提取要求：
{chr(10).join(instructions)}

严格返回JSON：
{{
  "Cargo forbidden": [],
  "Time": {{"min": null, "max": null, "unit": null}},
  "Volume Limit": {{"length_cm": null, "width_cm": null, "height_cm": null, "max_length_cm": null, "max_volume_m3": null, "formula": null, "raw": null}},
  "Volume to Weight parameter": null,
  "Pick&Packing/parcel": null,
  "Tax Policy": {{"delivery_term": null, "fob_limit_usd": null, "cif_limit_usd": null, "raw": null}}
}}
"""
    context = f"目标国家价格行：\n{country_context}\n\n价格使用说明：\n{note_text}\n\n计重规则：\n{weight_text}\n\n申报及税费：\n{tax_text}"
    return call_ai_json(prompt, context)


def ai_map_weight_rows(target_country: str, weight_rows: List[Dict[str, Any]], rule: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
目标国家：{target_country}

分析输入的真实Excel重量价格行。
{normalize_text(rule['AI指令'])}

固定目标重量max：
0.25、0.50、0.75、1.00、1.25、1.50、1.75、2.00、2.25、2.50、2.75、3.00。

严格返回：
{{
  "source_min_kg": null,
  "source_max_kg": null,
  "mapping": [
    {{"target_max_kg": 0.25, "source_excel_row": 12}}
  ]
}}

要求：
1. source_excel_row必须是输入中真实存在的Excel Row。
2. 每个target_max_kg选择实际覆盖该重量的源重量区间。
3. 超过源最大计费重量的target_max_kg返回null。
4. 不允许创造行号。
5. 仅依据输入，不得猜测。
"""
    return call_ai_json(prompt, json.dumps(weight_rows, ensure_ascii=False, indent=2))


# ============================================================
# 11. AI结果格式化
# ============================================================
def format_time(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, dict):
        mn, mx, unit = value.get("min"), value.get("max"), value.get("unit")
        if mn is None:
            return None
        return f"{mn} {unit}" if mx is None or mn == mx else f"{mn}~{mx} {unit}"
    return normalize_text(value)


def format_dimension(value: Any) -> Optional[str]:
    if not value:
        return None
    if not isinstance(value, dict):
        return normalize_text(value)
    parts = []
    if all(value.get(k) is not None for k in ["length_cm", "width_cm", "height_cm"]):
        parts.append(f"{value['length_cm']}×{value['width_cm']}×{value['height_cm']} cm")
    if value.get("max_length_cm") is not None:
        parts.append(f"max_length={value['max_length_cm']}cm")
    if value.get("max_volume_m3") is not None:
        parts.append(f"max_volume={value['max_volume_m3']}m³")
    if value.get("formula"):
        parts.append(f"formula={value['formula']}")
    return "; ".join(parts) or normalize_text(value.get("raw")) or None


def format_tax(value: Any) -> Optional[str]:
    if not value:
        return None
    if not isinstance(value, dict):
        return normalize_text(value)
    parts = []
    if value.get("delivery_term"):
        parts.append(str(value["delivery_term"]))
    if value.get("fob_limit_usd") is not None:
        parts.append(f"FOB < {value['fob_limit_usd']} USD")
    if value.get("cif_limit_usd") is not None:
        parts.append(f"CIF < {value['cif_limit_usd']} USD")
    return ", ".join(parts) or normalize_text(value.get("raw")) or None


def format_forbidden(value: Any) -> Optional[str]:
    if isinstance(value, list):
        return ", ".join(normalize_text(x) for x in value if normalize_text(x)) or None
    return normalize_text(value) or None


# ============================================================
# 12. 解析单个线路Sheet
# ============================================================
def parse_one_sheet(df: pd.DataFrame, sheet_name: str, target_country: str, rules: pd.DataFrame):
    rows, errors = [], []
    try:
        country_rule = get_rule(rules, "Destination Country")
        weight_rule = get_rule(rules, "Weight Range (max kg)")
        freight_rule = get_rule(rules, "RMB /kg")
        parcel_rule = get_rule(rules, "RMB /parcel")
        channel_id = extract_id(sheet_name)
        cargo = cargo_category(sheet_name)

        _, _, country_rows = find_country_rows(df, country_rule, target_country)
        if not country_rows:
            return rows, errors

        weight_col = find_cell(df, weight_rule["列定位类型"], weight_rule["列定位值"])
        freight_col = find_cell(df, freight_rule["列定位类型"], freight_rule["列定位值"])
        parcel_col = find_cell(df, parcel_rule["列定位类型"], parcel_rule["列定位值"])
        if not weight_col or not freight_col or not parcel_col:
            return rows, [{"Sheet": sheet_name, "Field": "Price Columns", "Error": "无法定位重量/运费/挂号费列"}]

        weight_source_rows = []
        country_context = []
        for r in country_rows:
            values = {f"Column_{c+1}": normalize_text(df.iat[r, c]) for c in range(df.shape[1]) if normalize_text(df.iat[r, c])}
            country_context.append({"Excel Row": r + 1, "Values": values})
            weight_raw = normalize_text(df.iat[r, weight_col[1]])
            if weight_raw:
                weight_source_rows.append({
                    "source_excel_row": r + 1,
                    "weight_range_raw": weight_raw,
                    "freight_raw": normalize_text(df.iat[r, freight_col[1]]),
                    "parcel_raw": normalize_text(df.iat[r, parcel_col[1]]),
                })

        if not weight_source_rows:
            return rows, [{"Sheet": sheet_name, "Field": "Weight Range", "Error": "目标国家没有找到重量价格行"}]

        metadata = ai_extract_metadata(
            target_country,
            json.dumps(country_context, ensure_ascii=False, indent=2),
            extract_section(df, "价格使用说明"),
            extract_section(df, "计重规则"),
            extract_section(df, "申报及税费"),
            rules,
        )

        weight_ai = ai_map_weight_rows(target_country, weight_source_rows, weight_rule)
        source_min, source_max = safe_float(weight_ai.get("source_min_kg")), safe_float(weight_ai.get("source_max_kg"))
        if source_min is None or source_max is None:
            return rows, [{"Sheet": sheet_name, "Field": "Weight Range", "Error": "AI无法确定源数据最小/最大计费重量"}]

        mapping = {}
        for item in weight_ai.get("mapping", []):
            mx = safe_float(item.get("target_max_kg"))
            src = item.get("source_excel_row")
            if mx is not None and src is not None:
                try:
                    mapping[round(mx, 2)] = int(src)
                except (ValueError, TypeError):
                    pass

        weight_steps = generate_weight_steps(source_min, source_max)
        pick_pack = safe_float(metadata.get("Pick&Packing/parcel"))
        volume_param = safe_float(metadata.get("Volume to Weight parameter"))

        for wmin, wmax in weight_steps:
            source_row = mapping.get(round(wmax, 2))
            if source_row is None:
                errors.append({"Sheet": sheet_name, "Field": "Weight Range", "Weight max": wmax, "Error": "AI没有指定对应源价格行"})
                continue

            source_idx = source_row - 1
            if source_idx not in country_rows:
                errors.append({"Sheet": sheet_name, "Field": "Weight Range", "Weight max": wmax, "Error": f"AI指定Row {source_row}不属于目标国家"})
                continue

            rkg = safe_float(df.iat[source_idx, freight_col[1]])
            rparcel = safe_float(df.iat[source_idx, parcel_col[1]])

            # 如果价格Cell不是纯数字，再让AI只解析这个Cell，不让AI决定最终价格逻辑
            if rkg is None:
                ai_price = call_ai_json('只从这个Cell文本提取RMB/kg数字，返回{"value":null}，没有明确数字就返回null。', normalize_text(df.iat[source_idx, freight_col[1]]))
                rkg = safe_float(ai_price.get("value"))
            if rparcel is None:
                ai_price = call_ai_json('只从这个Cell文本提取RMB/parcel数字，返回{"value":null}，没有明确数字就返回null。', normalize_text(df.iat[source_idx, parcel_col[1]]))
                rparcel = safe_float(ai_price.get("value"))

            total = None if any(v is None for v in [rkg, rparcel, pick_pack]) else round(wmax * rkg + rparcel + pick_pack, 2)

            rows.append({
                "ID": channel_id,
                "Destination Country": target_country,
                "Cargo Category": cargo,
                "Cargo forbidden": format_forbidden(metadata.get("Cargo forbidden")),
                "Time (workday/nature day)": format_time(metadata.get("Time")),
                "Volume Limit (cm)": format_dimension(metadata.get("Volume Limit")),
                "Volume to Weight parameter": volume_param,
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
# 13. 解析整个Excel：只解析用户输入的目标国家
# ============================================================
def parse_workbook(all_sheets: Dict[str, pd.DataFrame], target_country: str, rules: pd.DataFrame):
    id_rule = get_rule(rules, "ID")
    target_sheets = locate_sheets(all_sheets, id_rule)
    if not target_sheets:
        raise RuntimeError("没有找到符合当前供应商Mapping的线路Sheet。")

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
# 14. 新旧数据比较
# ============================================================
def get_country_worksheet(country: str):
    return get_google_worksheet(DATA_SHEET_ID, country, create=True)


def compare_data(new_df: pd.DataFrame, old_df: pd.DataFrame):
    if old_df.empty:
        return {"new": new_df.copy(), "updated": pd.DataFrame(), "unchanged": pd.DataFrame(), "final": new_df.copy()}

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
        changed = any(normalize_text(n[c]) != normalize_text(o[c]) for c in compare_cols)
        (updated_rows if changed else unchanged_rows).append(n.drop("_pk").to_dict())

    # 新文件中没有出现的旧记录暂时保留；不会误删
    untouched = old[~old["_pk"].isin(new["_pk"])].drop(columns="_pk", errors="ignore")
    final = pd.concat([untouched, pd.DataFrame(new_rows), pd.DataFrame(updated_rows), pd.DataFrame(unchanged_rows)], ignore_index=True)

    return {"new": pd.DataFrame(new_rows), "updated": pd.DataFrame(updated_rows), "unchanged": pd.DataFrame(unchanged_rows), "final": final}


def write_data(ws, df: pd.DataFrame):
    clean = df.fillna("")
    ws.clear()
    ws.update([clean.columns.tolist()] + clean.astype(str).values.tolist(), range_name="A1")
    return len(clean)


# ============================================================
# 15. App界面
# ============================================================
st.subheader("① 上传报价表")
uploaded_file = st.file_uploader("把供应商报价 Excel 拖到这里", type=["xlsx", "xls"])

st.subheader("② 输入目标国家/地区")
target_country = st.text_input("目标国家/地区", placeholder="例如：墨西哥、美国、加拿大").strip()

if uploaded_file:
    st.info(f"已选择文件：{uploaded_file.name}（{uploaded_file.size / 1024 / 1024:.1f} MB）")

run = st.button(
    "🚀 识别供应商并开始解析",
    type="primary",
    use_container_width=True,
    disabled=not uploaded_file or not target_country,
)

if run:
    try:
        with st.spinner("正在读取Excel..."):
            all_sheets = load_excel(uploaded_file.getvalue())

        supplier_code, supplier_name, mapping_sheet = detect_supplier(all_sheets)
        st.success(f"✅ 供应商：{supplier_name}（{supplier_code}）")
        st.info(f"✅ Mapping：{mapping_sheet}")

        rules = load_mapping(mapping_sheet)

        with st.spinner(f"正在解析目标国家：{target_country}"):
            parsed_df, errors_df = parse_workbook(all_sheets, target_country, rules)

        if parsed_df.empty:
            st.error(f"❌ 没有提取到【{target_country}】的数据。")
            if not errors_df.empty:
                st.dataframe(errors_df, use_container_width=True)
            st.stop()

        ws = get_country_worksheet(target_country)
        old_df = pd.DataFrame(ws.get_all_records())
        comparison = compare_data(parsed_df, old_df)

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
            st.dataframe(errors_df, use_container_width=True)

        st.download_button(
            "⬇️ 下载解析结果 CSV",
            parsed_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{supplier_code}_{target_country}.csv",
            mime="text/csv",
        )

        st.warning(f"确认后更新Google Sheet【{target_country}】；唯一键：ID + Destination Country + Weight Range (max kg)。")

        confirm = st.checkbox("我已检查预览结果，确认写入Google Sheet。")
        if st.button("✅ 确认并更新 Google Sheet", type="primary", use_container_width=True, disabled=not confirm):
            count = write_data(ws, comparison["final"])
            st.success(f"🎉 更新完成，共 {count} 条记录。")

    except Exception as e:
        st.error("❌ 运行失败")
        st.exception(e)
