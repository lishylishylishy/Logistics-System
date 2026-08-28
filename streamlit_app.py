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

# AI model API
AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
AI_MODEL = "gemini-3.1-flash-lite"

PRIMARY_KEYS = ["ID", "Destination Country", "Supplier", "Weight (kg)"]

STANDARD_FIELDS = [
    "ID", "Destination Country", "Supplier", "Cargo Category", "Cargo forbidden",
    "Time Min (day)", "Time Max (day)", "Time Type (workday/nature day)",
    "Volume Limit (cm)", "Volume to Weight Parameter",
    "Weight (kg)", "RMB /kg", "RMB /parcel",
    "Pick&Packing/parcel", "RMB in total", "DDP", "Extra Tax Required", "Tax Policy",
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


st.markdown(
    f"[规则库（Google Sheets）](https://docs.google.com/spreadsheets/d/{spreadsheet_key(RULE_SHEET_ID)}) ｜ "
    f"[目标数据表（Google Sheets）](https://docs.google.com/spreadsheets/d/{spreadsheet_key(DATA_SHEET_ID)})"
)


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
# ⑤ 供应商识别：一个供应商=一个tab，tab名即供应商名；命中后才加载校验
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def cached_mapping_sheet_names() -> List[str]:
    return get_mapping_sheet_names()


def mapping_tab_keywords(mapping_sheet: str) -> List[str]:
    # 轻量读取：只取"供应商识别关键词"列，不做全量schema校验，避免schema不全的tab报错
    try:
        ws = open_spreadsheet(RULE_SHEET_ID).worksheet(mapping_sheet)
        values = ws.get_all_values()
    except Exception:
        return []
    if not values:
        return []
    headers = [norm(x) for x in values[0]]
    if "供应商识别关键词" not in headers:
        return []
    idx = headers.index("供应商识别关键词")
    kws = []
    for row in values[1:]:
        if idx < len(row):
            kws.extend([x.strip() for x in re.split(r"[|,，;；]", norm(row[idx])) if x.strip()])
    return list(dict.fromkeys(kws))


def detect_supplier(file_name: str, all_sheets: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame, str]:
    candidates = []
    mapping_names = cached_mapping_sheet_names()

    for mapping_sheet in mapping_names:
        score = 0
        evidence = []
        tab_n = norm(mapping_sheet).lower()

        # 1) tab名即供应商名：tab名匹配文件名/报价表Sheet名
        if tab_n and tab_n in norm(file_name).lower():
            score += 100
            evidence.append(f"文件名:{mapping_sheet}")
        hit_sheets = [s for s in all_sheets if tab_n and tab_n in norm(s).lower()]
        if hit_sheets:
            score += 80 + min(len(hit_sheets), 20)
            evidence.append(f"Sheet:{hit_sheets[0]}")

        # 2) tab内"供应商识别关键词"辅助匹配
        for keyword in mapping_tab_keywords(mapping_sheet):
            keyword_n = keyword.lower()
            if keyword_n in norm(file_name).lower():
                score += 100
                evidence.append(f"文件名:{keyword}")
            hit_sheets = [s for s in all_sheets if keyword_n in norm(s).lower()]
            if hit_sheets:
                score += 80 + min(len(hit_sheets), 20)
                evidence.append(f"Sheet:{hit_sheets[0]}")

        if score:
            candidates.append((score, mapping_sheet, "；".join(evidence)))

    if not candidates:
        raise RuntimeError(
            "无法自动识别供应商。\n"
            f"当前规则库Mapping tab：{', '.join(mapping_names)}\n"
            f"上传文件：{file_name}\n"
            "规则库中一个供应商对应一个tab，tab名即供应商名（如4PX）；"
            "请确保tab名或tab内“供应商识别关键词”能匹配文件名/Sheet名。"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RuntimeError(f"供应商识别冲突：{candidates[0][1]} / {candidates[1][1]}；请修改tab名或供应商识别关键词。")

    winner = candidates[0][1]
    rules = load_mapping(winner)
    return winner, rules, candidates[0][2]


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


def mapping_section_anchors(rules: pd.DataFrame) -> List[str]:
    # 章节锚点全部来自Mapping的行定位值（text_anchor），py不硬编码任何供应商章节名
    anchors = []
    for _, row in rules.iterrows():
        if norm(row.get("行定位类型")) == "text_anchor":
            anchors.extend(a for a in norm(row.get("行定位值")).split("|") if a)
    return list(dict.fromkeys(anchors))


def find_section_start(df: pd.DataFrame, anchors: List[str]) -> int:
    for r in range(len(df)):
        text = " ".join(norm(v) for v in df.iloc[r].tolist() if norm(v))
        if any(a in text for a in anchors):
            return r
    return len(df)


def row_has_data(df: pd.DataFrame, r: int) -> bool:
    return any(norm(v) for v in df.iloc[r].tolist())


def find_country_rows(df: pd.DataFrame, country_rule: Dict[str, Any], target_country: str, anchors: List[str]) -> List[int]:
    country_col = locate_column(df, country_rule)
    header_hit = locate_header_cell(df, country_rule)
    header_row = header_hit[0] if header_hit else 0
    end_row = find_section_start(df, anchors)
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


@register_parser("parser_mapping_tab_name")
def parser_mapping_tab_name(value: Any, rule: Dict[str, Any]) -> Optional[str]:
    # 供应商名=规则库Mapping的tab名，由引擎传入
    return norm(value) or None


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


@register_parser("parse_weight_segment")
def parse_weight_segment(value: Any, rule: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    # 规则表参数示例：{"pattern": "([\\d.]+)\\s*[<＜]\\s*W\\s*[≤<=]+\\s*([\\d.]+)"}
    params = json_params(rule)
    pattern = params.get("pattern")
    if not pattern:
        raise RuntimeError(f"字段【{rule.get('字段')}】的parse_weight_segment缺少pattern参数")
    try:
        m = re.search(pattern, norm(value))
    except re.error as e:
        raise RuntimeError(f"字段【{rule.get('字段')}】的parse_weight_segment正则错误：{pattern}；{e}") from e
    if not m or len(m.groups()) < 2:
        return None
    return float(m.group(1)), float(m.group(2))


@register_parser("generate_weight_ladder")
def generate_weight_ladder(source_max: Optional[float], rule: Dict[str, Any]) -> List[float]:
    # 规则表参数示例：{"step_kg": 0.25, "ladder_max_kg": 3.0}
    # 梯度点不得大于该ID+国家的最大计费重量（源重量段并集上界）
    if source_max is None:
        return []
    params = json_params(rule)
    step = float(params.get("step_kg", 0.25))
    ladder_max = float(params.get("ladder_max_kg", 3.0))
    cap = min(ladder_max, source_max)
    ladder, w = [], step
    while w <= cap + 1e-9:
        ladder.append(round(w, 2))
        w += step
    return ladder


def run_parser(rule: Dict[str, Any], *args) -> Any:
    name = norm(rule.get("Python解析器", ""))
    if not name:
        return None
    if name not in PARSERS:
        raise RuntimeError(f"字段【{rule.get('字段')}】引用了py中不存在的解析器：{name}")
    return PARSERS[name](*args, rule)


ALLOWED_FORMULA_VARS = {"weight", "rmb_kg", "rmb_parcel", "pick_pack_calc"}


def calculate_total_by_mapping(rule: Dict[str, Any], weight: float, rmb_kg: Optional[float], rmb_parcel: Optional[float], pick_pack_display: Any) -> Optional[float]:
    if rmb_kg is None or rmb_parcel is None or weight is None:
        return None

    pick_pack_calc = safe_float(pick_pack_display)
    pick_pack_calc = 0.0 if pick_pack_calc is None else pick_pack_calc

    params = json_params(rule)
    expression = norm(params.get("formula"))

    if not expression:
        return round(weight * rmb_kg + rmb_parcel + pick_pack_calc, 2)

    values = {
        "weight": weight,
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
        "Cargo forbidden", "Time Min (day)", "Time Max (day)", "Time Type (workday/nature day)",
        "Volume Limit (cm)", "Volume to Weight Parameter", "Pick&Packing/parcel",
        "DDP", "Extra Tax Required", "Tax Policy",
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
  "Time": {{"min": null, "max": null, "unit": null, "risk_note": null}},
  "Volume Limit": {{"length_cm": null, "width_cm": null, "height_cm": null, "max_length_cm": null, "max_summary_of_3_lengths_cm": null, "raw": null}},
  "Volume to Weight Parameter": null,
  "Pick&Packing/parcel": "unknown",
  "DDP": "unknown",
  "Extra Tax Required": "unknown",
  "Tax Policy": {{"delivery_term": null, "fob_limit_usd": null, "cif_limit_usd": null, "tax_formula": null, "tax_payer": null, "raw": null}}
}}

要求：
- Time.min/Time.max只输出数字，单一值时两者相同；unit只能是workday或nature day；risk_note为延误风险提示原文，没有则null。
- DDP、Extra Tax Required只能是yes/no/unknown，且严格按照各自字段要求中的判定规则执行。
- Pick&Packing不能确认费用时必须返回"unknown"。
- 不得猜测。
"""

    context = f"目标国家价格/服务行：\n{country_context}\n\n"
    for field, text in notes.items():
        if norm(text):
            context += f"{field} 相关章节：\n{text}\n\n"

    return ai_json(prompt, context)


# ============================================================
# ⑩ AI结果格式化
# ============================================================
def format_dimension(value: Any) -> Optional[str]:
    if not value:
        return None
    if not isinstance(value, dict):
        return norm(value) or None
    parts = []
    # 1. 长×宽×高
    if all(
        value.get(x) is not None
        for x in ["length_cm", "width_cm", "height_cm"]
    ):
        parts.append(
            f"max size = "
            f"{value['length_cm']}×"
            f"{value['width_cm']}×"
            f"{value['height_cm']}cm"
        )
    # 2. 单边最大长度
    if value.get("max_length_cm") is not None:
        parts.append(
            f"max lenth = {value['max_length_cm']}cm"
        )
    # 3. 三边之和最大值
    if value.get("max_summary_of_3_lengths_cm") is not None:
        parts.append(
            f"max summary of 3 lenthes = "
            f"{value['max_summary_of_3_lengths_cm']}cm"
        )
    # 都没结构化成功时保留原文
    return "; ".join(parts) or norm(value.get("raw")) or None

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
def parse_route(df: pd.DataFrame, sheet_name: str, target_country: str, rules: pd.DataFrame, supplier: str) -> Dict[str, Any]:
    route: Dict[str, Any] = {"sheet": sheet_name, "country": target_country, "errors": []}

    # 线路/供应商粒度字段：ID、Cargo Category从sheet名解析，Supplier=Mapping tab名
    for field in ["ID", "Cargo Category", "Supplier"]:
        rule = get_rule(rules, field)
        source = sheet_name if norm(rule["原始提取类型"]) == "sheet_name" else supplier
        try:
            route[field] = run_parser(rule, source)
        except Exception as e:
            route[field] = None
            route["errors"].append(f"{field}: {e}")

    # 国家数据行（章节锚点来自Mapping）
    country_rule = get_rule(rules, "Destination Country")
    anchors = mapping_section_anchors(rules)
    try:
        crows = find_country_rows(df, country_rule, target_country, anchors)
    except Exception as e:
        route["errors"].append(f"国家行定位失败: {e}")
        return route
    if not crows:
        route["not_serviced"] = True
        route["errors"].append(f"Sheet【{sheet_name}】中未找到国家【{target_country}】的数据行")
        return route

    country_context = "\n".join(
        f"Excel Row {r + 1}: " + " | ".join(norm(v) for v in df.iloc[r].tolist() if norm(v))
        for r in crows
    )

    # 国家粒度、列定位字段：时效/尺寸原始文本（供AI上下文与兜底）
    try:
        col = locate_column(df, get_rule(rules, "Time Min (day)"))
        route["Time_raw"] = " | ".join(dict.fromkeys(norm(df.iat[r, col]) for r in crows if norm(df.iat[r, col])))
    except Exception as e:
        route["Time_raw"] = ""
        route["errors"].append(f"Time: {e}")
    try:
        col = locate_column(df, get_rule(rules, "Volume Limit (cm)"))
        route["Volume Limit (cm)_raw"] = " | ".join(dict.fromkeys(norm(df.iat[r, col]) for r in crows if norm(df.iat[r, col])))
    except Exception as e:
        route["Volume Limit (cm)_raw"] = ""
        route["errors"].append(f"Volume Limit (cm): {e}")

    # 章节文本字段（供AI，锚点来自Mapping行定位值）
    notes = {}
    for field in ["Cargo forbidden", "Volume to Weight Parameter", "Pick&Packing/parcel", "DDP", "Tax Policy"]:
        rule = get_rule(rules, field)
        if norm(rule["行定位类型"]) in {"text_anchor", "header_text"}:
            notes[field] = extract_section(df, norm(rule["行定位值"]))

    # 源重量段（中间字段，不写入最终数据表）
    segments = []
    try:
        seg_rule = get_rule(rules, "源重量段")
        kg_rule = get_rule(rules, "RMB /kg")
        parcel_rule = get_rule(rules, "RMB /parcel")
        wcol = locate_column(df, seg_rule)
        kgcol = locate_column(df, kg_rule)
        pcol = locate_column(df, parcel_rule)
        for r in crows:
            bounds = run_parser(seg_rule, df.iat[r, wcol])
            if bounds is None:
                route["errors"].append(f"第{r + 1}行重量段无法解析：{norm(df.iat[r, wcol])}")
                continue
            segments.append({
                "excel_row": r + 1,
                "min": bounds[0],
                "max": bounds[1],
                "RMB /kg": run_parser(kg_rule, df.iat[r, kgcol]),
                "RMB /parcel": run_parser(parcel_rule, df.iat[r, pcol]),
            })
    except Exception as e:
        route["errors"].append(f"源重量段解析失败: {e}")

    # AI结构化（AI指令全部来自Mapping）
    meta: Dict[str, Any] = {}
    try:
        meta = ai_metadata(target_country, country_context, notes, rules)
    except Exception as e:
        route["errors"].append(f"AI结构化失败: {e}")

    pick_pack = meta.get("Pick&Packing/parcel")
    route["Cargo forbidden"] = format_forbidden(meta.get("Cargo forbidden"))

    # 时效拆三列：最小天数/最大天数/类型与延误风险；AI失败时用原始文本兜底
    t = meta.get("Time")
    t = t if isinstance(t, dict) else {}
    t_min, t_max, unit, risk = t.get("min"), t.get("max"), t.get("unit"), t.get("risk_note")
    raw_time = norm(route.get("Time_raw"))
    if t_min is None and raw_time:
        m = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)", raw_time)
        if m:
            t_min, t_max = float(m.group(1)), float(m.group(2))
        else:
            m2 = re.search(r"\d+(?:\.\d+)?", raw_time)
            t_min = float(m2.group()) if m2 else None
        if not unit:
            unit = "nature day" if "自然日" in raw_time else "workday"
        if not risk and "延误" in raw_time:
            risk = raw_time
    route["Time Min (day)"] = t_min
    route["Time Max (day)"] = t_max if t_max is not None else t_min
    route["Time Type (workday/nature day)"] = "; ".join(str(x) for x in [unit or "workday", risk] if norm(x)) or None

    route["Volume Limit (cm)"] = format_dimension(meta.get("Volume Limit")) or route.get("Volume Limit (cm)_raw") or None
    route["Volume to Weight Parameter"] = format_volume_weight(meta.get("Volume to Weight Parameter"))
    route["Pick&Packing/parcel"] = pick_pack if pick_pack not in {None, ""} else "unknown"
    route["DDP"] = meta.get("DDP") or "unknown"
    route["Extra Tax Required"] = meta.get("Extra Tax Required") or "unknown"
    route["Tax Policy"] = format_tax(meta.get("Tax Policy"))

    # Weight (kg) 梯度展开：0.25递增、不超过最大计费重量，每点继承所在源重量段价格
    weight_records = []
    try:
        ladder_rule = get_rule(rules, "Weight (kg)")
        total_rule = get_rule(rules, "RMB in total")
        if segments:
            source_max = max(s["max"] for s in segments)
            for w in run_parser(ladder_rule, source_max):
                seg = next((s for s in segments if s["min"] < w <= s["max"]), None)
                if seg is None:
                    route["errors"].append(f"Weight (kg)={w} 未落入任何源重量段")
                    continue
                total = calculate_total_by_mapping(total_rule, w, seg["RMB /kg"], seg["RMB /parcel"], pick_pack)
                weight_records.append({
                    "source_excel_row": seg["excel_row"],
                    "Weight (kg)": w,
                    "RMB /kg": seg["RMB /kg"],
                    "RMB /parcel": seg["RMB /parcel"],
                    "RMB in total": total,
                })
    except Exception as e:
        route["errors"].append(f"Weight (kg)解析失败: {e}")

    route["weight_records"] = weight_records
    route["notes"] = notes
    return route


# ============================================================
# ⑫ Streamlit 页面逻辑
# ============================================================
uploaded = st.file_uploader("上传供应商报价表（xlsx）", type=["xlsx", "xls"])
target_country = st.text_input("目标国家（如：美国）", "").strip()

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
                    route = parse_route(all_sheets[sheet_name], sheet_name, target_country, rules, supplier)
                results.append(route)

            ok_routes = [r for r in results if r.get("weight_records")]
            st.write(f"成功解析 {len(ok_routes)}/{len(results)} 条线路")

            for route in ok_routes:
                with st.expander(f"{route['ID']} | {route['sheet']} | {route['country']}", expanded=True):
                    st.json({k: v for k, v in route.items() if k not in {"notes", "errors"}}, expanded=False)
                    if route["errors"]:
                        st.warning("；".join(route["errors"]))

            failed = [r for r in results if not r.get("weight_records")]
            not_serviced = [r for r in failed if r.get("not_serviced")]
            real_failed = [r for r in failed if not r.get("not_serviced")]
            if not_serviced:
                with st.expander(f"未开通【{target_country}】的线路（{len(not_serviced)}个，正常跳过）"):
                    st.write("、".join(norm(r.get("sheet")) for r in not_serviced))
            if real_failed:
                with st.expander(f"解析失败的线路（{len(real_failed)}个）"):
                    for r in real_failed:
                        st.write(f"- {r['sheet']}: {'；'.join(r['errors'])}")

            if ok_routes and st.button("写入目标数据表（Google Sheets）"):
                ws = get_country_worksheet(target_country)

                def fmt_num(x: Any) -> Any:
                    if x is None or x == "":
                        return ""
                    try:
                        f = float(x)                # 如果传了 decimals（比如 2），就强制保留指定位数的小数
                        if decimals is not None:    # 没传 decimals，则保持原样（整数转 int，小数保留原样）
                            return round(f, decimals)
                        return str(int(f)) if f == int(f) else str(f)
                    except (TypeError, ValueError):
                        return norm(x)

                # 批量写入+主键去重：避免逐行append触发Google写配额(429)，重跑不产生重复行
                raw = ws.get_all_values()
                # 过滤全空行并记录真实行号（兼容首行空行/gspread空行填充）
                numbered = [(idx, row) for idx, row in enumerate(raw, start=1) if any(norm(x) for x in row)]

                def col_letter(n: int) -> str:
                    s = ""
                    while n:
                        n, r = divmod(n - 1, 26)
                        s = chr(65 + r) + s
                    return s

                if not numbered:
                    ws.append_row(STANDARD_FIELDS, value_input_option="RAW")
                    final_header = [str(x) for x in STANDARD_FIELDS]
                    data_rows: List[Tuple[int, List[str]]] = []
                else:
                    first_no, first_row = numbered[0]
                    first_names = [norm(x) for x in first_row]
                    is_header = first_no == 1 and sum(1 for k in PRIMARY_KEYS if k in first_names) >= len(PRIMARY_KEYS) - 1
                    if is_header:
                        final_header = first_names
                        data_rows = numbered[1:]
                        # 表头自动补齐：缺失的标准列名补到行末
                        missing = [f for f in STANDARD_FIELDS if f not in final_header]
                        if missing:
                            a = len(final_header) + 1
                            ws.update(f"{col_letter(a)}1:{col_letter(a + len(missing) - 1)}1", [missing], value_input_option="RAW")
                            final_header = final_header + missing
                    else:
                        # 无表头：插入标准表头行，数据行号整体+1
                        ws.insert_row([str(x) for x in STANDARD_FIELDS], 1, value_input_option="RAW")
                        final_header = [str(x) for x in STANDARD_FIELDS]
                        data_rows = [(no + 1, row) for no, row in numbered]

                # 列数据与表头名绑定：按表头位置写入，用户移动列不影响正确性
                pos = {name: i for i, name in enumerate(final_header)}
                ncols = len(final_header)
                std_cols = sorted(pos[f] for f in STANDARD_FIELDS)
                runs: List[List[int]] = []
                for c in std_cols:
                    if runs and c == runs[-1][1] + 1:
                        runs[-1][1] = c
                    else:
                        runs.append([c, c])

                updates, appends = [], []
                if all(k in pos for k in PRIMARY_KEYS):
                    old = {}
                    for no, row in data_rows:
                        key = tuple(norm(row[pos[k]]) if pos[k] < len(row) else "" for k in PRIMARY_KEYS)
                        old.setdefault(key, no)
                    for route in ok_routes:
                        for rec in route["weight_records"]:
                            rec_dict = {
                                "ID": norm(route.get("ID")),
                                "Destination Country": target_country,
                                "Supplier": norm(route.get("Supplier")),
                                "Cargo Category": norm(route.get("Cargo Category")),
                                "Cargo forbidden": norm(route.get("Cargo forbidden")),
                                "Time Min (day)": fmt_num(route.get("Time Min (day)")),
                                "Time Max (day)": fmt_num(route.get("Time Max (day)")),
                                "Time Type (workday/nature day)": norm(route.get("Time Type (workday/nature day)")),
                                "Volume Limit (cm)": norm(route.get("Volume Limit (cm)")),
                                "Volume to Weight Parameter": norm(route.get("Volume to Weight Parameter")),
                                #以下5项指定保留 2 位小数
                                "Weight (kg)": fmt_num(rec["Weight (kg)"],2),
                                "RMB /kg": fmt_num(rec["RMB /kg"],2),
                                "RMB /parcel": fmt_num(rec["RMB /parcel"],2),
                                "Pick&Packing/parcel": norm(route.get("Pick&Packing/parcel")),
                                "RMB in total": fmt_num(rec["RMB in total"],2),
                                
                                "DDP": norm(route.get("DDP")),
                                "Extra Tax Required": norm(route.get("Extra Tax Required")),
                                "Tax Policy": norm(route.get("Tax Policy")),
                            }
                            row_values = [""] * ncols
                            for f in STANDARD_FIELDS:
                                row_values[pos[f]] = rec_dict[f]
                            key = tuple(rec_dict[k] for k in PRIMARY_KEYS)
                            if key in old:
                                no = old[key]
                                for a, b in runs:
                                    updates.append({
                                        "range": f"{col_letter(a + 1)}{no}:{col_letter(b + 1)}{no}",
                                        "values": [row_values[a:b + 1]],
                                    })
                            else:
                                old[key] = -1
                                appends.append(row_values)

                if updates:
                    ws.batch_update(updates, value_input_option="USER_ENTERED")
                for i in range(0, len(appends), 1000):
                    ws.append_rows(appends[i:i + 1000], value_input_option="USER_ENTERED")
                st.success(f"写入完成：更新{len(updates)}行、新增{len(appends)}行（共{len(updates) + len(appends)}行）")

    except Exception as e:
        st.error(str(e))
