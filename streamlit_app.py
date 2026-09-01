import io, json, re
from typing import Any, Dict, List, Optional

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI

# ============================================================
# ① 固定配置：沿用原App的 Secrets、Google Sheet 和 Gemini 连接
# ============================================================
RULE_SHEET_ID = st.secrets["RULE_SHEET_ID"]       # 供应商Mapping规则库
DATA_SHEET_ID = st.secrets["DATA_SHEET_ID"]       # 最终数据表
AI_API_KEY = st.secrets["API_KEY"]                # Gemini API Key
GCP_JSON = st.secrets["gcp_json"]                 # Google Service Account JSON
AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
AI_MODEL = "gemini-3.1-flash-lite"

# 唯一键：同一ID+国家+重量只保留一行
PRIMARY_KEYS = ["ID", "Destination Country", "Weight (kg)"]

# 最终数据表字段；Weight由Python生成，两个Total由Google Sheets公式生成，AI不参与计算
STANDARD_FIELDS = [
    "ID", "Destination Country", "Supplier", "Cargo Category", "Cargo forbidden",
    "Time Min (day)", "Time Max (day)", "Time Type (workday/nature day)",
    "Volume Limit (cm)", "Volume to Weight Parameter", "Weight (kg)", "RMB /kg",
    "RMB /parcel", "Pick&Packing/parcel", "RMB in total", "USD in total",
    "DDP", "Extra Tax Required", "Tax Policy",
]

st.set_page_config(page_title="物流报价解析系统", page_icon="📦", layout="wide")
st.title("📦 物流报价解析系统")
st.caption("文件名识别供应商 → 取得对应Mapping → AI按自然语言规则提取 → Python/Google Sheets计算 → 更新数据")

# ============================================================
# ② 基础工具
# ============================================================
def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(v).replace("\u3000", " ")).strip()


def spreadsheet_key(v: str) -> str:
    v = norm(v)
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", v)
    return m.group(1) if m else v


def safe_float(v: Any) -> Optional[float]:
    text = norm(v).replace(",", "").replace("，", "")
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def fmt_num(v: Any, decimals: Optional[int] = None) -> Any:
    if v is None or v == "":
        return ""
    try:
        f = float(v)
        return round(f, decimals) if decimals is not None else (str(int(f)) if f == int(f) else str(f))
    except (TypeError, ValueError):
        return norm(v)


def col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# ============================================================
# ③ Google Sheets：只做三件事
#    1. 连接原来的Google Sheet
#    2. 根据文件名找到供应商Mapping Tab
#    3. 把命中的Mapping原文交给AI，不解释Mapping
# ============================================================
@st.cache_resource
def gsheet_client():
    info = json.loads(GCP_JSON, strict=False)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    return gspread.authorize(Credentials.from_service_account_info(info, scopes=scopes))


def open_ss(value: str):
    try:
        return gsheet_client().open_by_key(spreadsheet_key(value))
    except Exception as e:
        raise RuntimeError(f"打开Google Spreadsheet失败：{e}") from e


@st.cache_data(ttl=600, show_spinner=False)
def get_mapping_names() -> List[str]:
    result = []
    for ws in open_ss(RULE_SHEET_ID).worksheets():
        name = norm(ws.title)
        if name.lower() in {"supplier_config", "config"}:
            continue
        values = ws.get_all_values()
        if values and "字段" in [norm(x) for x in values[0]]:
            result.append(name)
    return result


def get_mapping_keywords(mapping_sheet: str) -> List[str]:
    # 这里只读取供应商识别关键词；不解析其他Mapping规则
    try:
        values = open_ss(RULE_SHEET_ID).worksheet(mapping_sheet).get_all_values()
    except Exception:
        return [mapping_sheet]
    if not values:
        return [mapping_sheet]
    headers = [norm(x) for x in values[0]]
    if "供应商识别关键词" not in headers:
        return [mapping_sheet]
    idx = headers.index("供应商识别关键词")
    kws = [mapping_sheet]
    for row in values[1:]:
        if idx < len(row):
            kws.extend(x.strip() for x in re.split(r"[|,，;；]", norm(row[idx])) if x.strip())
    return list(dict.fromkeys(kws))


def detect_supplier(file_name: str) -> str:
    # 保留原逻辑：只用文件名匹配Mapping Tab名/供应商识别关键词，不让AI选择供应商
    name = norm(file_name).lower()
    matches = []
    for mapping in get_mapping_names():
        for kw in get_mapping_keywords(mapping):
            k = norm(kw).lower()
            if k and k in name:
                matches.append((len(k), mapping, kw))
    if not matches:
        raise RuntimeError(
            f"无法根据文件名识别供应商：{file_name}\n"
            f"当前Mapping Tabs：{', '.join(get_mapping_names())}\n"
            "请在对应Mapping Tab的“供应商识别关键词”中加入文件名中的供应商关键词。"
        )
    matches.sort(key=lambda x: x[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0] and matches[0][1] != matches[1][1]:
        raise RuntimeError(f"供应商识别冲突：{matches[0][1]} / {matches[1][1]}")
    return matches[0][1]


def load_mapping(mapping_sheet: str) -> str:
    # Mapping按原文读取；自然语言怎么改，Python都不解释、不校验规则内容
    values = open_ss(RULE_SHEET_ID).worksheet(mapping_sheet).get_all_values()
    if not values:
        raise RuntimeError(f"Mapping【{mapping_sheet}】为空")
    lines = [f"===== Mapping Tab: {mapping_sheet} ====="]
    for row_no, row in enumerate(values, 1):
        cells = [norm(x) for x in row]
        if any(cells):
            lines.append(f"Row {row_no}: " + " | ".join(cells))
    return "\n".join(lines)


# ============================================================
# ④ Excel：完整读取，AI自己按Mapping自然语言寻找Sheet/行/列
# ============================================================
@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)


def excel_text(all_sheets: Dict[str, pd.DataFrame]) -> str:
    blocks = []
    for name, df in all_sheets.items():
        lines = []
        for r in range(len(df)):
            vals = [norm(v) for v in df.iloc[r].tolist() if norm(v)]
            if vals:
                lines.append(f"Excel Row {r + 1}: " + " | ".join(vals))
        if lines:
            blocks.append(f"===== Sheet: {name} =====\n" + "\n".join(lines))
    return "\n\n".join(blocks)


# ============================================================
# ⑤ Python固定重量：AI只能填写价格，不得修改重量
# ============================================================
def fixed_weights() -> List[float]:
    return [round(i * 0.25, 2) for i in range(1, 13)]


# ============================================================
# ⑥ AI：Mapping是唯一业务规则，AI负责理解Excel和提取信息
# ============================================================
@st.cache_data(show_spinner=False, max_entries=200)
def ai_json(prompt: str, context: str) -> Dict[str, Any]:
    client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
    res = client.chat.completions.create(
        model=AI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "你是严谨的物流报价Excel数据提取专家。只能依据上传Excel和给定Mapping提取，不得使用自己的物流规则补充、推断或延伸价格。找不到或无法确认时返回null；Mapping明确要求unknown时返回unknown。必须返回合法JSON，不要输出任何解释。"},
            {"role": "user", "content": f"{prompt}\n\n{context}"},
        ],
    )
    raw = res.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.I)
    return json.loads(raw)


def extract_routes(file_name: str, supplier: str, country: str, mapping: str, excel: str, weights: List[float]) -> List[Dict[str, Any]]:
    prompt = f"""
上传文件名称：{file_name}
当前供应商：{supplier}
目标国家：{country}

【唯一有效的Mapping规则】
{mapping}

【必须遵守】
1. 当前供应商已经由Python根据文件名确定，不得重新选择供应商，不得调用其他供应商Mapping。
2. Mapping中的自然语言是唯一的数据提取规则；严格按照Mapping执行。
3. Mapping会告诉你去哪张Sheet、找什么关键词/表头/章节、怎么判断以及输出什么格式。你可以查看全部Sheet，但不能用自己的物流经验替代Mapping。
4. 只提取目标国家“{country}”。
5. Python已经固定生成这些Weight：{json.dumps(weights, ensure_ascii=False)}。必须完整返回这些Weight，不能新增、删除或修改Weight。
6. 对每个Weight，根据Mapping寻找实际存在且覆盖该Weight的源报价重量段，提取RMB /kg和RMB /parcel。
7. 如果Excel中不存在能够覆盖该Weight的实际报价重量段，RMB /kg和RMB /parcel必须返回null；不得把相邻重量段价格延伸过去，不得估算、平均、插值或猜测。
8. AI不得计算RMB in total，也不得计算USD in total。
9. 除Weight、RMB in total、USD in total外，其余字段都按Mapping提取。
10. 只返回JSON，不要解释。

JSON结构：
{{
  "routes": [
    {{
      "sheet": null,
      "ID": null,
      "Cargo Category": null,
      "Cargo forbidden": null,
      "Time Min (day)": null,
      "Time Max (day)": null,
      "Time Type (workday/nature day)": null,
      "Volume Limit (cm)": null,
      "Volume to Weight Parameter": null,
      "Pick&Packing/parcel": "unknown",
      "DDP": "unknown",
      "Extra Tax Required": "unknown",
      "Tax Policy": null,
      "Weight Prices": [
        {{"Weight (kg)": 0.25, "RMB /kg": null, "RMB /parcel": null}}
      ]
    }}
  ]
}}

Weight Prices必须完整返回：{json.dumps([{"Weight (kg)": w, "RMB /kg": None, "RMB /parcel": None} for w in weights], ensure_ascii=False)}
"""
    result = ai_json(prompt, f"【Excel原始数据】\n{excel}")
    routes = result.get("routes", [])
    if not isinstance(routes, list):
        raise RuntimeError("AI返回的routes不是列表")

    for route in routes:
        route["Supplier"] = supplier
        raw_prices = route.get("Weight Prices", [])
        price_map = {safe_float(x.get("Weight (kg)")): x for x in raw_prices if isinstance(x, dict) and safe_float(x.get("Weight (kg)")) is not None}
        route["Weight Prices"] = [price_map.get(w, {"Weight (kg)": w, "RMB /kg": None, "RMB /parcel": None}) for w in weights]
    return routes


# ============================================================
# ⑦ AI结果只做轻量格式整理；不重新解释业务规则
# ============================================================
def format_dimension(v: Any) -> str:
    if not isinstance(v, dict):
        return norm(v)
    parts = []
    if all(v.get(x) is not None for x in ["length_cm", "width_cm", "height_cm"]):
        parts.append(f"max size = {v['length_cm']}×{v['width_cm']}×{v['height_cm']}cm")
    if v.get("max_length_cm") is not None:
        parts.append(f"max lenth = {v['max_length_cm']}cm")
    if v.get("max_summary_of_3_lengths_cm") is not None:
        parts.append(f"max summary of 3 lenthes = {v['max_summary_of_3_lengths_cm']}cm")
    return "; ".join(parts) or norm(v.get("raw"))


# ============================================================
# ⑧ 写入Google Sheets
#    Python不计算Total；直接写入Google Sheets公式
#    公式按字段名称找真实列，不写死K/L/M/O
# ============================================================
def get_country_ws(country: str):
    ss = open_ss(DATA_SHEET_ID)
    try:
        return ss.worksheet(country)
    except gspread.exceptions.WorksheetNotFound:
        return ss.add_worksheet(title=country, rows=2000, cols=30)


def build_total_formula(row_no: int, pos: Dict[str, int]) -> str:
    w = col_letter(pos["Weight (kg)"] + 1)
    kg = col_letter(pos["RMB /kg"] + 1)
    parcel = col_letter(pos["RMB /parcel"] + 1)
    pack = col_letter(pos["Pick&Packing/parcel"] + 1)
    return f'=IF(OR({w}{row_no}="",{kg}{row_no}=""),"",{w}{row_no}*{kg}{row_no}+IFERROR(VALUE({parcel}{row_no}),0)+IFERROR(VALUE({pack}{row_no}),0))'


def build_usd_formula(row_no: int, pos: Dict[str, int]) -> str:
    rmb_total = col_letter(pos["RMB in total"] + 1)
    return f'=IF({rmb_total}{row_no}="","",{rmb_total}{row_no}*GOOGLEFINANCE("CURRENCY:CNYUSD"))'


def prepare_header(ws):
    raw = ws.get_all_values()
    numbered = [(i, r) for i, r in enumerate(raw, 1) if any(norm(x) for x in r)]

    if not numbered:
        ws.append_row(STANDARD_FIELDS, value_input_option="RAW")
        return STANDARD_FIELDS, []

    first_no, first = numbered[0]
    first = [norm(x) for x in first]
    is_header = first_no == 1 and sum(k in first for k in PRIMARY_KEYS) >= len(PRIMARY_KEYS) - 1

    if is_header:
        header, data = first, numbered[1:]
    else:
        ws.insert_row(STANDARD_FIELDS, 1, value_input_option="RAW")
        header, data = STANDARD_FIELDS, [(n + 1, r) for n, r in numbered]

    missing = [f for f in STANDARD_FIELDS if f not in header]
    if missing:
        start = len(header) + 1
        ws.update(f"{col_letter(start)}1:{col_letter(start + len(missing) - 1)}1", [missing], value_input_option="RAW")
        header += missing
    return header, data


def write_rows(country: str, routes: List[Dict[str, Any]]):
    ws = get_country_ws(country)
    header, data = prepare_header(ws)
    pos = {name: i for i, name in enumerate(header)}
    old = {}

    for no, row in data:
        old[tuple(norm(row[pos[k]]) if pos[k] < len(row) else "" for k in PRIMARY_KEYS)] = no

    # 只更新标准字段所在的连续区域，保留数据表里的其他自定义列
    std_cols = sorted(pos[f] for f in STANDARD_FIELDS)
    runs = []
    for c in std_cols:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])

    last_row = max([n for n, _ in data], default=1)
    updates, appends = [], []

    for route in routes:
        for rec in route.get("Weight Prices", []):
            weight = safe_float(rec.get("Weight (kg)"))
            if weight is None:
                continue

            row = {
                "ID": norm(route.get("ID")), "Destination Country": country, "Supplier": norm(route.get("Supplier")),
                "Cargo Category": norm(route.get("Cargo Category")), "Cargo forbidden": norm(route.get("Cargo forbidden")),
                "Time Min (day)": fmt_num(route.get("Time Min (day)")), "Time Max (day)": fmt_num(route.get("Time Max (day)")),
                "Time Type (workday/nature day)": norm(route.get("Time Type (workday/nature day)")),
                "Volume Limit (cm)": format_dimension(route.get("Volume Limit (cm)")),
                "Volume to Weight Parameter": norm(route.get("Volume to Weight Parameter")),
                "Weight (kg)": fmt_num(weight, 2), "RMB /kg": fmt_num(rec.get("RMB /kg"), 2),
                "RMB /parcel": fmt_num(rec.get("RMB /parcel"), 2), "Pick&Packing/parcel": norm(route.get("Pick&Packing/parcel")) or "unknown",
                "RMB in total": "", "USD in total": "", "DDP": norm(route.get("DDP")) or "unknown",
                "Extra Tax Required": norm(route.get("Extra Tax Required")) or "unknown", "Tax Policy": norm(route.get("Tax Policy")),
            }

            key = tuple(row[k] for k in PRIMARY_KEYS)
            row_no = old.get(key)
            if row_no is None or row_no == -1:
                last_row += 1
                row_no = last_row
                values = [row.get(f, "") for f in header]
                values[pos["RMB in total"]] = build_total_formula(row_no, pos)
                values[pos["USD in total"]] = build_usd_formula(row_no, pos)
                appends.append(values)
                old[key] = -1
            else:
                values = [row.get(f, "") for f in header]
                values[pos["RMB in total"]] = build_total_formula(row_no, pos)
                values[pos["USD in total"]] = build_usd_formula(row_no, pos)
                for a, b in runs:
                    updates.append({"range": f"{col_letter(a + 1)}{row_no}:{col_letter(b + 1)}{row_no}", "values": [values[a:b + 1]]})

    if updates:
        ws.batch_update(updates, value_input_option="USER_ENTERED")
    if appends:
        ws.append_rows(appends, value_input_option="USER_ENTERED")
    return len(updates), len(appends)


# ============================================================
# ⑨ 页面主流程
# ============================================================
st.markdown(f"[规则库（Google Sheets）](https://docs.google.com/spreadsheets/d/{spreadsheet_key(RULE_SHEET_ID)}) ｜ [目标数据表（Google Sheets）](https://docs.google.com/spreadsheets/d/{spreadsheet_key(DATA_SHEET_ID)})")

uploaded = st.file_uploader("上传供应商报价表（xlsx）", type=["xlsx", "xls"])
target_country = st.text_input("目标国家（如：美国）", "").strip()

if uploaded is not None:
    try:
        file_bytes = uploaded.getvalue()
        all_sheets = load_excel(file_bytes)
        supplier = detect_supplier(uploaded.name)
        mapping = load_mapping(supplier)

        st.success(f"识别供应商：{supplier}")
        st.write(f"Mapping：{supplier} ｜ Excel Sheet：{len(all_sheets)} 个")

        if not target_country:
            st.info("请输入目标国家后开始解析。")
        else:
            weights = fixed_weights()
            excel = excel_text(all_sheets)
            with st.spinner("AI正在严格按照当前供应商Mapping读取Excel……"):
                routes = extract_routes(uploaded.name, supplier, target_country, mapping, excel, weights)

            # 只要AI返回线路且重量点完整，就允许预览；价格为null的重量也保留
            ok_routes = [r for r in routes if r.get("Weight Prices")]
            st.write(f"AI解析线路：{len(ok_routes)} 条")

            for route in ok_routes:
                with st.expander(f"{norm(route.get('ID'))} | {norm(route.get('sheet'))} | {target_country}", expanded=True):
                    st.json(route, expanded=False)

            if ok_routes and st.button("写入目标数据表（Google Sheets）"):
                updated, added = write_rows(target_country, ok_routes)
                st.success(f"写入完成：更新 {updated} 行，新增 {added} 行")
    except Exception as e:
        st.error(str(e))
