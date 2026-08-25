import ast
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
# ① 固定配置：这里才放真正"通用"的系统配置
# ============================================================
RULE_SHEET_ID = st.secrets["RULE_SHEET_ID"]       # 供应商 Mapping 规则库
DATA_SHEET_ID = st.secrets["DATA_SHEET_ID"]       # 最终数据
AI_API_KEY = st.secrets["API_KEY"]                # AI API
GCP_JSON = st.secrets["gcp_json"]                 # Google Service Account JSON

# BUG修复：原代码URL被反引号包住，会变成非法字符串
AI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
AI_MODEL = "qwen3.7-plus"

PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]

# 通用业务规则：0~3kg，每0.25kg一个梯度
STANDARD_WEIGHTS = [
    (0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00),
    (1.00, 1.25), (1.25, 1.50), (1.50, 1.75), (1.75, 2.00),
    (2.00, 2.25), (2.25, 2.50), (2.50, 2.75), (2.75, 3.00),
]

STANDARD_FIELDS = [
    "ID", "Destination Country", "Cargo Category", "Cargo forbidden",
    "Time (workday/nature day)", "Volume Limit (cm)", "Volume to Weight parameter",
    "Weight Range (min kg)", "Weight Range (max kg)", "RMB /kg", "RMB /parcel",
    "Pick&Packing/parcel", "RMB in total", "Tax Policy",
]


# ============================================================
# ② 页面
# ============================================================
st.set_page_config(page_title="物流报价解析系统", page_icon="📦", layout="wide")
st.title("📦 物流报价解析系统")
st.caption("上传报价表 → 自动识别供应商 → 加载对应 Mapping → 输入目标国家 → 解析 → 预览 → 更新")


# ============================================================
# ③ 基础工具
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


def spreadsheet_key(value: str) -> str:
    value = norm(value)
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", value)
    return m.group(1) if m else value


def to_bool(value: Any) -> bool:
    return norm(value).lower() in {"true", "1", "yes", "y", "是", "启用"}


def safe_float(value: Any) -> Optional[float]:
    text = norm(value).replace(",", "").replace("，", "")
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def json_params(rule: Dict[str, Any]) -> Dict[str, Any]:
    text = norm(rule.get("Python规则参数", ""))
    if not text:
        return {}
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Python规则参数必须是JSON对象")
        return value
    except Exception as e:
        raise RuntimeError(f"字段【{rule.get('字段')}】的Python规则参数不是合法JSON：{e}") from e


# ============================================================
# ④ Google Sheets
# ============================================================
@st.cache_resource
def get_gsheet_client():
    info = json.loads(GCP_JSON, strict=False)
    # BUG修复：原代码scopes两个URL被反引号包住，授权必失败
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def open_spreadsheet(value: str):
    key = spreadsheet_key(value)
    try:
        return get_gsheet_client().open_by_key(key)
    except Exception as e:
        raise RuntimeError(f"打开Google Spreadsheet失败：{key}\n{e}") from e


def get_mapping_sheet_names() -> List[str]:
    result = []
    for ws in open_spreadsheet(RULE_SHEET_ID).worksheets():
        name = norm(ws.title)
        if name.lower() in {"supplier_config", "config"}:
            continue
        values = ws.get_all_values()
        if not values:
            continue
        headers = [norm(x) for x in values[0]]
        if "字段" in headers:
            result.append(name)
    return result


def load_mapping(mapping_sheet: str) -> pd.DataFrame:
    sh = open_spreadsheet(RULE_SHEET_ID)
    try:
        ws = sh.worksheet(mapping_sheet)
    except gspread.exceptions.WorksheetNotFound as e:
        raise RuntimeError(f"规则库中不存在供应商Mapping：{mapping_sheet}") from e

    df = pd.DataFrame(ws.get_all_records())
    required = [
        "字段", "是否AI读取", "提取粒度", "记录唯一键",
        "Sheet定位类型", "Sheet定位值", "行定位类型", "行定位值",
        "列定位类型", "列定位值", "原始提取类型",
        "Python解析器", "AI指令", "是否必填",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Mapping【{mapping_sheet}】缺少列：{', '.join(missing)}")

    if "Python规则参数" not in df.columns:
        df["Python规则参数"] = ""
    if "供应商识别关键词" not in df.columns:
        df["供应商识别关键词"] = ""

    missing_fields = [f for f in STANDARD_FIELDS if f not in set(df["字段"].astype(str).str.strip())]
    if missing_fields:
        raise RuntimeError(f"Mapping【{mapping_sheet}】缺少标准字段：{', '.join(missing_fields)}")

    return df


def get_country_worksheet(country: str):
    sh = open_spreadsheet(DATA_SHEET_ID)
    try:
        return sh.worksheet(country)
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=country, rows=2000, cols=30)


# ============================================================
# ⑤ 供应商识别：先识别，再加载该供应商Mapping
# ============================================================
def mapping_detection_keywords(mapping_sheet: str, rules: pd.DataFrame) -> List[str]:
    raw_values = [norm(x) for x in rules.get("供应商识别关键词", pd.Series(dtype=str)).tolist() if norm(x)]
    values = []
    for raw in raw_values:
        values.extend([x.strip() for x in re.split(r"[|,，;；]", raw) if x.strip()])
    return list(dict.fromkeys(values)) if values else [mapping_sheet]


@st.cache_data(show_spinner=False, ttl=600)
def cached_mapping_sheet_names() -> List[str]:
    return get_mapping_sheet_names()


def detect_supplier(file_name: str, all_sheets: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame, str]:
    candidates = []
    # BUG修复：原代码每个候选供应商都重复调两次get_mapping_sheet_names()（每次都读云端表）
    mapping_names = cached_mapping_sheet_names()

    for mapping_sheet in mapping_names:
        rules = load_mapping(mapping_sheet)
        keywords = mapping_detection_keywords(mapping_sheet, rules)
        score = 0
        evidence = []

        for keyword in keywords:
            keyword_n = norm(keyword).lower()
            if not keyword_n:
                continue

            if keyword_n in norm(file_name).lower():
                score += 100
                evidence.append(f"文件名:{keyword}")

            hit_sheets = [s for s in all_sheets if keyword_n in norm(s).lower()]
            if hit_sheets:
                score += 80 + min(len(hit_sheets), 20)
                evidence.append(f"Sheet:{hit_sheets[0]}")

        if score:
            candidates.append((score, mapping_sheet, rules, "；".join(evidence)))

    if not candidates:
        raise RuntimeError(
            "无法自动识别供应商。\n"
            f"当前规则库供应商Mapping：{', '.join(mapping_names)}\n"
            f"上传文件：{file_name}\n"
            "请在供应商Mapping增加“供应商识别关键词”，或确保文件名/Sheet名包含供应商名称。"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RuntimeError(f"供应商识别冲突：{candidates[0][1]} / {candidates[1][1]}；请增加供应商识别关键词。")

    return candidates[0][1], candidates[0][2], candidates[0][3]


# ============================================================
# ⑥ Excel
# ============================================================
@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)


# ============================================================
# ⑦ Mapping定位引擎
# ============================================================
def get_rule(rules: pd.DataFrame, field: str) -> Dict[str, Any]:
    hit = rules[rules["字段"].astype(str).str.strip() == field]
    if hit.empty:
        raise RuntimeError(f"Mapping中没有字段：{field}")
    return hit.iloc[0].to_dict()


def text_match(text: str, rule_type: str, rule_value: str) -> bool:
    text, value = norm(text), norm(rule_value)
    if rule_type == "exact":
        return text == value
    if rule_type == "contains":
        return value in text
    if rule_type == "regex":
        try:
            return bool(re.search(value, text, re.I))
        except re.error as e:
            raise RuntimeError(f"Mapping正则错误：{value}；{e}") from e
    if rule_type == "none":
        return True
    return False


def locate_sheets(all_sheets: Dict[str, pd.DataFrame], rule: Dict[str, Any]) -> List[str]:
    return [name for name in all_sheets if text_match(name, rule["Sheet定位类型"], rule["Sheet定位值"])]


def locate_header_cell(df: pd.DataFrame, rule: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    col_type = norm(rule["列定位类型"])
    col_value = norm(rule["列定位值"])
    row_type = norm(rule["行定位类型"])
    row_value = norm(rule["行定位值"])

    if col_type == "none":
        return None

    for r in range(min(len(df), 30)):
        row_text = " | ".join(norm(v) for v in df.iloc[r].tolist() if norm(v))
        if row_type in {"header_text", "text_anchor"} and row_value and row_value not in row_text:
            continue

        for c in range(df.shape[1]):
            value = norm(df.iat[r, c])
            ok = (
                (col_type == "exact_header" and value == col_value)
                or (col_type == "contains_header" and col_value in value)
                or (col_type == "exact_text" and value == col_value)
                or (col_type == "contains_text" and col_value in value)
            )
            if ok:
                return r, c

    for r in range(len(df)):
        for c in range(df.shape[1]):
            value = norm(df.iat[r, c])
            if col_type == "exact_header" and value == col_value:
                return r, c
            if col_type == "contains_header" and col_value in value:
                return r, c

    return None


def locate_column(df: pd.DataFrame, rule: Dict[str, Any]) -> int:
    hit = locate_header_cell(df, rule)
    if not hit:
        raise RuntimeError(f"无法定位列：{rule['列定位类型']} / {rule['列定位值']}")
    return hit[1]


def find_section_start(df: pd.DataFrame) -> int:
    anchors = ["价格使用说明", "计重规则", "申报及税费"]
    for r in range(len(df)):
        text = " ".join(norm(v) for v in df.iloc[r].tolist() if norm(v))
        if any(a in text for a in anchors):
            return r
    return len(df)


def row_has_data(df: pd.DataFrame, r: int) -> bool:
    return any(norm(v) for v in df.iloc[r].tolist())


def find_country_rows(df: pd.DataFrame, country_rule: Dict[str, Any], target_country: str) -> List[int]:
    country_col = locate_column(df, country_rule)
    header_hit = locate_header_cell(df, country_rule)
    header_row = header_hit[0] if header_hit else 0
    end_row = find_section_start(df)
    target = norm(target_country)

    # BUG修复：原代码会把国家名之后的空行也计入（国家名向下延续），导致后续取数取到NaN
    rows, current_country = [], ""
    for r in range(header_row + 1, end_row):
        value = norm(df.iat[r, country_col])
        if value:
            current_country = value
        if current_country == target and row_has_data(df, r):
            rows.append(r)
    return rows


SECTION_TITLE = re.compile(r"^[一二三四五六七八九十百]+、")


def extract_section(df: pd.DataFrame, anchor: str) -> str:
    # BUG修复：原代码stops写死3个章节名，规则表里"计重规则|计费重量"这类多锚点会被截断；
    # 改为：锚点支持 | 分隔多个候选，结束以下一个"X、"章节标题为准
    anchors = [a for a in norm(anchor).split("|") if a]
    start = None
    for r in range(len(df)):
        text = " ".join(norm(v) for v in df.iloc[r].tolist() if norm(v))
        if any(a in text for a in anchors):
            start = r
            break
    if start is None:
        return ""

    lines = []
    for r in range(start, len(df)):
        vals = [norm(v) for v in df.iloc[r].tolist() if norm(v)]
        if not vals:
            continue
        text = " | ".join(vals)
        if r > start and SECTION_TITLE.match(vals[0]) and not any(a in vals[0] for a in anchors):
            break
        lines.append(f"Excel Row {r + 1}: {text}")
    return "\n".join(lines)


# ============================================================
# ⑧ Python通用解析器：函数本体通用，专用参数全部来自规则表
# ============================================================
PARSERS: Dict[str, Any] = {}


def register_parser(name: str):
    def deco(fn):
        PARSERS[name] = fn
        return fn
    return deco


@register_parser("parser_sheet_name_regex_group")
def parser_sheet_name_regex_group(sheet_name: str, rule: Dict[str, Any]) -> Optional[str]:
    pattern = norm(rule["Sheet定位值"])
    group = int(json_params(rule).get("group", 1))
    try:
        match = re.search(pattern, sheet_name, re.I)
    except re.error as e:
        raise RuntimeError(f"字段【{rule.get('字段')}】Sheet定位正则错误：{pattern}；{e}") from e
    return match.group(group).strip() if match and len(match.groups()) >= group else None


@register_parser("parser_sheet_keyword_map")
def parser_sheet_keyword_map(sheet_name: str, rule: Dict[str, Any]) -> Optional[str]:
    params = json_params(rule)
    hits = []
    for result, keywords in params.get("keyword_map", {}).items():
        if any(norm(k) and norm(k) in sheet_name for k in keywords):
            hits.append(result)
    if not hits:
        return params.get("default")
    return params.get("sep", ", ").join(hits) if params.get("multi") else hits[0]


@register_parser("parse_number")
def parse_number(value: Any, rule: Dict[str, Any]) -> Optional[float]:
    return safe_float(value)


@register_parser("parse_weight_bound")
def parse_weight_bound(value: Any, rule: Dict[str, Any]) -> Optional[float]:
    # 规则表参数示例：{"pattern": "([\\d.]+)\\s*[<＜]\\s*W\\s*[≤<=]+\\s*([\\d.]+)", "group": 1}
    params = json_params(rule)
    pattern = params.get("pattern")
    if not pattern:
        raise RuntimeError(f"字段【{rule.get('字段')}】的parse_weight_bound缺少pattern参数")
    try:
        m = re.search(pattern, norm(value))
    except re.error as e:
        raise RuntimeError(f"字段【{rule.get('字段')}】的parse_weight_bound正则错误：{pattern}；{e}") from e
    if not m:
        return None
    group = int(params.get("group", 1))
    return float(m.group(group)) if len(m.groups()) >= group and m.group(group) else None


def run_parser(rule: Dict[str, Any], *args) -> Any:
    name = norm(rule.get("Python解析器", ""))
    if not name:
        return None
    if name not in PARSERS:
        raise RuntimeError(f"字段【{rule.get('字段')}】引用了py中不存在的解析器：{name}")
    return PARSERS[name](*args, rule)


def generate_standard_weight_ranges(source_min: float, source_max: float) -> List[Tuple[float, float]]:
    if source_min is None or source_max is None or source_min >= source_max:
        return []

    result = []
    for smin, smax in STANDARD_WEIGHTS:
        if smax <= source_min or smin >= source_max:
            continue
        wmin = max(smin, source_min)
        wmax = min(smax, source_max)
        if wmax > wmin:
            result.append((round(wmin, 2), round(wmax, 2)))
    return result


ALLOWED_FORMULA_VARS = {"max_weight", "rmb_kg", "rmb_parcel", "pick_pack_calc"}


def calculate_total_by_mapping(rule: Dict[str, Any], max_weight: float, rmb_kg: Optional[float], rmb_parcel: Optional[float], pick_pack_display: Any) -> Optional[float]:
    if rmb_kg is None or rmb_parcel is None or max_weight is None:
        return None

    pick_pack_calc = safe_float(pick_pack_display)
    pick_pack_calc = 0.0 if pick_pack_calc is None else pick_pack_calc

    params = json_params(rule)
    expression = norm(params.get("formula"))

    if not expression:
        return round(max_weight * rmb_kg + rmb_parcel + pick_pack_calc, 2)

    values = {
        "max_weight": max_weight,
        "rmb_kg": rmb_kg,
        "rmb_parcel": rmb_parcel,
        "pick_pack_calc": pick_pack_calc,
    }

    try:
        tree = ast.parse(expression, mode="eval")

        def ev(node):
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Name) and node.id in ALLOWED_FORMULA_VARS:
                return float(values[node.id])
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                v = ev(node.operand)
                return v if isinstance(node.op, ast.UAdd) else -v
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                a, b = ev(node.left), ev(node.right)
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if isinstance(node.op, ast.Div):
                    return a / b
            raise ValueError("公式包含不允许的操作")

        return round(ev(tree), 2)
    except Exception:
        return None


# ============================================================
# ⑨ AI
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
                "content": "你是严谨的物流报价表数据提取专家。只能依据输入，不得猜测；无法确认时返回unknown或null；必须返回合法JSON。",
            },
            {"role": "user", "content": f"{prompt}\n\n原始数据：\n{context}"},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def ai_metadata(target_country: str, country_context: str, notes: Dict[str, str], rules: pd.DataFrame) -> Dict[str, Any]:
    fields = [
        "Cargo forbidden", "Time (workday/nature day)", "Volume Limit (cm)",
        "Volume to Weight parameter", "Pick&Packing/parcel", "Tax Policy",
    ]

    instructions = []
    for field in fields:
        rule = get_rule(rules, field)
        if to_bool(rule["是否AI读取"]):
            instructions.append(f"{field}: {norm(rule['AI指令'])}")

    prompt = f"""
目标国家：{target_country}
只提取该目标国家。

字段要求：
{chr(10).join(instructions)}

严格返回JSON：
{{
  "Cargo forbidden": [],
  "Time": {{"min": null, "max": null, "unit": null}},
  "Volume Limit": {{"length_cm": null, "width_cm": null, "height_cm": null, "max_length_cm": null, "max_volume_m3": null, "formula": null, "raw": null}},
  "Volume to Weight parameter": null,
  "Pick&Packing/parcel": "unknown",
  "Tax Policy": {{"delivery_term": null, "fob_limit_usd": null, "cif_limit_usd": null, "raw": null}}
}}

要求：
- Time严格按照Mapping中的AI指令返回单位。
- Pick&Packing不能确认费用时必须返回"unknown"。
- 不得猜测。
"""

    context = (
        f"目标国家价格/服务行：\n{country_context}\n\n"
        f"价格使用说明：\n{notes.get('价格使用说明', '')}\n\n"
        f"计重规则：\n{notes.get('计重规则', '')}\n\n"
        f"申报及税费：\n{notes.get('申报及税费', '')}"
    )

    return ai_json(prompt, context)


def ai_weight_rows(target_country: str, weight_rows: List[Dict[str, Any]], rule: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
目标国家：{target_country}
分析下面真实存在的Excel重量价格行。

{norm(rule["AI指令"])}

标准target_max_kg只能是：
0.25、0.50、0.75、1.00、1.25、1.50、1.75、2.00、2.25、2.50、2.75、3.00。

严格返回：
{{
  "source_min_kg": null,
  "source_max_kg": null,
  "mapping": [
    {{"target_max_kg":0.25,"source_excel_row":12}}
  ]
}}

要求：
1. source_excel_row必须来自输入中的真实Excel Row。
2. target_max_kg选择能够覆盖该重量的源重量区间。
3. 超过source_max_kg的target_max_kg返回null。
4. 不得创造行号。
"""

    return ai_json(prompt, json.dumps(weight_rows, ensure_ascii=False, indent=2))


# ============================================================
# ⑩ AI结果格式化
# ============================================================
def format_time(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, dict):
        mn, mx, unit = value.get("min"), value.get("max"), value.get("unit")
        if mn is None:
            return None
        unit = unit or "workday"
        return f"{mn} {unit}" if mx is None or mn == mx else f"{mn}~{mx} {unit}"
    return norm(value) or None


def format_dimension(value: Any) -> Optional[str]:
    if not value:
        return None
    if not isinstance(value, dict):
        return norm(value) or None

    parts = []
    if all(value.get(x) is not None for x in ["length_cm", "width_cm", "height_cm"]):
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
    if value is None:
        return None
    if isinstance(value, list):
        items = [norm(x) for x in value if norm(x)]
        return "；".join(items) if items else None
    return norm(value) or None


def format_volume_weight(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    return norm(value) or None


# ============================================================
# ⑪ 单条线路解析主流程（通用引擎，专用规则全部来自Mapping）
# ============================================================
def parse_route(df: pd.DataFrame, sheet_name: str, target_country: str, rules: pd.DataFrame) -> Dict[str, Any]:
    route: Dict[str, Any] = {"sheet": sheet_name, "country": target_country, "errors": []}

    # 线路粒度字段：ID、Cargo Category（从sheet名解析）
    for field in ["ID", "Cargo Category"]:
        rule = get_rule(rules, field)
        try:
            route[field] = run_parser(rule, sheet_name)
        except Exception as e:
            route[field] = None
            route["errors"].append(f"{field}: {e}")

    # 国家数据行
    country_rule = get_rule(rules, "Destination Country")
    try:
        crows = find_country_rows(df, country_rule, target_country)
    except Exception as e:
        route["errors"].append(f"国家行定位失败: {e}")
        return route
    if not crows:
        route["errors"].append(f"Sheet【{sheet_name}】中未找到国家【{target_country}】的数据行")
        return route

    # 国家粒度、列定位字段：Time、Volume Limit
    for field, fmt in [("Time (workday/nature day)", None), ("Volume Limit (cm)", None)]:
        rule = get_rule(rules, field)
        try:
            col = locate_column(df, rule)
            route[f"{field}_raw"] = " | ".join(dict.fromkeys(norm(df.iat[r, col]) for r in crows if norm(df.iat[r, col])))
        except Exception as e:
            route[f"{field}_raw"] = ""
            route["errors"].append(f"{field}: {e}")

    # 章节文本字段（供AI）
    notes = {}
    for field in ["Cargo forbidden", "Volume to Weight parameter", "Pick&Packing/parcel", "Tax Policy"]:
        rule = get_rule(rules, field)
        if norm(rule["行定位类型"]) in {"text_anchor", "header_text"}:
            notes[field] = extract_section(df, norm(rule["行定位值"]))

    # 重量段明细
    weight_records = []
    try:
        wmin_rule = get_rule(rules, "Weight Range (min kg)")
        wmax_rule = get_rule(rules, "Weight Range (max kg)")
        kg_rule = get_rule(rules, "RMB /kg")
        parcel_rule = get_rule(rules, "RMB /parcel")
        total_rule = get_rule(rules, "RMB in total")

        wcol = locate_column(df, wmin_rule)
        kgcol = locate_column(df, kg_rule)
        pcol = locate_column(df, parcel_rule)

        for r in crows:
            wmin = run_parser(wmin_rule, df.iat[r, wcol])
            wmax = run_parser(wmax_rule, df.iat[r, wcol])
            rmb_kg = run_parser(kg_rule, df.iat[r, kgcol])
            rmb_parcel = run_parser(parcel_rule, df.iat[r, pcol])
            total = calculate_total_by_mapping(total_rule, wmax, rmb_kg, rmb_parcel, None)
            weight_records.append({
                "excel_row": r + 1,
                "Weight Range (min kg)": wmin,
                "Weight Range (max kg)": wmax,
                "RMB /kg": rmb_kg,
                "RMB /parcel": rmb_parcel,
                "RMB in total": total,
            })
    except Exception as e:
        route["errors"].append(f"重量段解析失败: {e}")

    route["weight_records"] = weight_records
    route["notes"] = notes
    return route


# ============================================================
# ⑫ Streamlit 页面逻辑
# ============================================================
uploaded = st.file_uploader("上传供应商报价表（xlsx）", type=["xlsx", "xls"])
target_country = st.text_input("目标国家（如：墨西哥）", "").strip()

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    try:
        all_sheets = load_excel(file_bytes)
        supplier, rules, evidence = detect_supplier(uploaded.name, all_sheets)
        st.success(f"识别供应商：{supplier}（依据：{evidence}）")

        route_sheets = locate_sheets(all_sheets, get_rule(rules, "ID"))
        st.write(f"命中线路Sheet {len(route_sheets)} 个：{', '.join(route_sheets)}")

        if not target_country:
            st.info("请输入目标国家后开始解析。")
        else:
            results = []
            for sheet_name in route_sheets:
                with st.spinner(f"解析 {sheet_name} ..."):
                    route = parse_route(all_sheets[sheet_name], sheet_name, target_country, rules)
                results.append(route)

            ok_routes = [r for r in results if r.get("weight_records")]
            st.write(f"成功解析 {len(ok_routes)}/{len(results)} 条线路")

            for route in ok_routes:
                with st.expander(f"{route['ID']} | {route['sheet']} | {route['country']}", expanded=True):
                    st.json({k: v for k, v in route.items() if k not in {"notes", "errors"}}, expanded=False)
                    if route["errors"]:
                        st.warning("；".join(route["errors"]))

            failed = [r for r in results if not r.get("weight_records")]
            if failed:
                with st.expander(f"未解析出重量段的线路（{len(failed)}）"):
                    for r in failed:
                        st.write(f"- {r['sheet']}: {'；'.join(r['errors']) or '无数据行'}")

            if ok_routes and st.button("写入目标数据表（Google Sheets）"):
                ws = get_country_worksheet(target_country)
                existing = ws.get_all_values()
                if not existing:
                    ws.append_row(STANDARD_FIELDS, value_input_option="RAW")
                written = 0
                for route in ok_routes:
                    for rec in route["weight_records"]:
                        row_values = [
                            norm(route.get("ID")),
                            target_country,
                            norm(route.get("Cargo Category")),
                            "",  # Cargo forbidden（AI，待接入）
                            norm(route.get("Time (workday/nature day)_raw")),
                            norm(route.get("Volume Limit (cm)_raw")),
                            "",  # Volume to Weight parameter（AI，待接入）
                            rec["Weight Range (min kg)"] if rec["Weight Range (min kg)"] is not None else "",
                            rec["Weight Range (max kg)"] if rec["Weight Range (max kg)"] is not None else "",
                            rec["RMB /kg"] if rec["RMB /kg"] is not None else "",
                            rec["RMB /parcel"] if rec["RMB /parcel"] is not None else "",
                            "",  # Pick&Packing（AI，待接入）
                            rec["RMB in total"] if rec["RMB in total"] is not None else "",
                            "",  # Tax Policy（AI，待接入）
                        ]
                        ws.append_row(row_values, value_input_option="RAW")
                        written += 1
                st.success(f"已写入 {written} 行到数据表【{target_country}】工作表")

    except Exception as e:
        st.error(str(e))
