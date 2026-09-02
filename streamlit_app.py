import io
import json
import re
import time
import tempfile
import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import gspread
import pandas as pd
import requests
import streamlit as st
from google.oauth2.service_account import Credentials


# ============================================================
# ① 固定配置：沿用原 App 的 Secrets / Google Sheet / Gemini 配置
# ============================================================
RULE_SHEET_ID = st.secrets["RULE_SHEET_ID"]
DATA_SHEET_ID = st.secrets["DATA_SHEET_ID"]
AI_API_KEY = st.secrets["API_KEY"]
GCP_JSON = st.secrets["gcp_json"]
AI_MODEL = "gemini-3.1-flash-lite"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_UPLOAD = "https://generativelanguage.googleapis.com/upload/v1beta/files"

# 最终数据唯一键：同一ID、国家、重量点只保留一行
PRIMARY_KEYS = ["ID", "Destination Country", "Weight (kg)"]

# 最终数据表固定字段；Weight / RMB in total / USD in total由Python或公式控制
STANDARD_FIELDS = [
    "ID", "Destination Country", "Supplier", "Cargo Category", "Cargo forbidden",
    "Time Min (day)", "Time Max (day)", "Time Type (workday/nature day)",
    "Volume Limit (cm)", "Volume to Weight Parameter", "Weight (kg)", "RMB /kg",
    "RMB /parcel", "Pick&Packing/parcel", "RMB in total", "USD in total",
    "DDP", "Extra Tax Required", "Tax Policy",
]

# Python固定提供给AI的重量点；AI不得修改、不增不减
FIXED_WEIGHTS = [round(i * 0.25, 2) for i in range(1, 13)]


# ============================================================
# ② 页面
# ============================================================
st.set_page_config(page_title="物流报价解析系统", page_icon="📦", layout="wide")
st.title("📦 物流报价解析系统")
st.caption("上传报价表 → 文件名识别供应商 → 读取对应Mapping → Gemini完整阅读Excel → 预览 → 写入")


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


def safe_float(value: Any) -> Optional[float]:
    text = norm(value).replace(",", "").replace("，", "")
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def fmt_num(value: Any, decimals: Optional[int] = None) -> Any:
    if value is None or value == "":
        return ""
    try:
        f = float(value)
        if decimals is not None:
            return round(f, decimals)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return norm(value)


def parse_json_response(text: str) -> Dict[str, Any]:
    raw = norm(text)
    raw = re.sub(r"^```json\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


st.markdown(
    f"[规则库（Google Sheets）](https://docs.google.com/spreadsheets/d/{spreadsheet_key(RULE_SHEET_ID)}) ｜ "
    f"[目标数据表（Google Sheets）](https://docs.google.com/spreadsheets/d/{spreadsheet_key(DATA_SHEET_ID)})"
)


# ============================================================
# ④ Google Sheets：完全沿用原连接
# ============================================================
@st.cache_resource
def get_gsheet_client():
    info = json.loads(GCP_JSON, strict=False)
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
        values = ws.get_all_values()
        if not values:
            continue
        headers = [norm(x) for x in values[0]]
        if "字段" in headers:
            result.append(norm(ws.title))
    return result


@st.cache_data(show_spinner=False, ttl=600)
def cached_mapping_sheet_names() -> List[str]:
    return get_mapping_sheet_names()


def mapping_tab_keywords(mapping_sheet: str) -> List[str]:
    """只读取供应商识别关键词，用于Python按文件名识别供应商。"""
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
            kws.extend(x.strip() for x in re.split(r"[|,，;；]", norm(row[idx])) if x.strip())
    return list(dict.fromkeys(kws))


def load_mapping_text(mapping_sheet: str) -> str:
    """读取命中的供应商Mapping原文；Python不解释规则，只原样交给AI。"""
    try:
        ws = open_spreadsheet(RULE_SHEET_ID).worksheet(mapping_sheet)
        values = ws.get_all_values()
    except gspread.exceptions.WorksheetNotFound as e:
        raise RuntimeError(f"规则库中不存在供应商Mapping：{mapping_sheet}") from e
    if not values:
        raise RuntimeError(f"Mapping【{mapping_sheet}】为空")
    lines = []
    for i, row in enumerate(values, start=1):
        cells = [norm(v) for v in row]
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            lines.append(f"Row {i}: " + " | ".join(cells))
    return "\n".join(lines)


def detect_supplier(file_name: str) -> Tuple[str, str]:
    """只根据上传文件名匹配Mapping Tab名称或供应商识别关键词。"""
    name = norm(file_name).lower()
    hits = []
    for mapping_sheet in cached_mapping_sheet_names():
        score = 0
        evidence = []
        tab = norm(mapping_sheet).lower()
        if tab and tab in name:
            score += 100
            evidence.append(f"文件名命中Tab:{mapping_sheet}")
        for keyword in mapping_tab_keywords(mapping_sheet):
            if norm(keyword).lower() in name:
                score += 100
                evidence.append(f"文件名命中关键词:{keyword}")
        if score:
            hits.append((score, mapping_sheet, "；".join(evidence)))
    if not hits:
        raise RuntimeError(f"无法根据文件名识别供应商：{file_name}\n当前Mapping Tab：{', '.join(cached_mapping_sheet_names())}")
    hits.sort(key=lambda x: x[0], reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        raise RuntimeError(f"供应商识别冲突：{hits[0][1]} / {hits[1][1]}；请修改Mapping中的供应商识别关键词。")
    return hits[0][1], hits[0][2]


def get_country_worksheet(country: str):
    sh = open_spreadsheet(DATA_SHEET_ID)
    try:
        return sh.worksheet(country)
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=country, rows=2000, cols=30)


# ============================================================
# ⑤ Excel：Python保留读取，只用于读取/统计，不再把全文转成Prompt
# ============================================================
@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)


# ============================================================
# ⑥ Gemini Files API：完整Excel文件直接交给Gemini
# ============================================================
def gemini_upload_file(file_bytes: bytes, file_name: str) -> Dict[str, Any]:
    """使用同一个Gemini API Key上传完整XLSX，不把Excel转成超长文本。"""
    meta = {"file": {"display_name": file_name}}
    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(len(file_bytes)),
        "X-Goog-Upload-Header-Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "Content-Type": "application/json",
    }
    r = requests.post(f"{GEMINI_UPLOAD}?key={AI_API_KEY}", headers=headers, json=meta, timeout=60)
    r.raise_for_status()
    upload_url = r.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise RuntimeError("Gemini Files API未返回上传地址")
    upload_headers = {
        "Content-Length": str(len(file_bytes)),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
        "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    r = requests.post(upload_url, headers=upload_headers, data=file_bytes, timeout=180)
    r.raise_for_status()
    data = r.json().get("file") or r.json()
    return data


def gemini_wait_file(file_name: str, timeout: int = 120) -> Dict[str, Any]:
    """等待Gemini把上传的XLSX处理到可供模型读取。"""
    end = time.time() + timeout
    while time.time() < end:
        r = requests.get(f"{GEMINI_BASE}/{file_name}?key={AI_API_KEY}", timeout=30)
        r.raise_for_status()
        data = r.json()
        state = norm((data.get("state") or {}).get("name"))
        if state in {"ACTIVE", ""}:
            return data
        if state == "FAILED":
            raise RuntimeError(f"Gemini文件处理失败：{data}")
        time.sleep(2)
    raise RuntimeError("Gemini处理Excel文件超时")


def gemini_delete_file(file_name: str) -> None:
    try:
        requests.delete(f"{GEMINI_BASE}/{file_name}?key={AI_API_KEY}", timeout=30)
    except Exception:
        pass


@st.cache_data(show_spinner=False, max_entries=100)
def ai_extract_full_excel(file_hash: str, file_bytes: bytes, file_name: str, supplier: str, mapping_text: str, target_country: str) -> Dict[str, Any]:
    """一次上传完整Excel，一次让Gemini按Mapping处理目标国家。"""
    file_info = gemini_upload_file(file_bytes, file_name)
    file_name_api = file_info.get("name")
    file_uri = file_info.get("uri")
    mime_type = file_info.get("mimeType") or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if not file_name_api or not file_uri:
        raise RuntimeError(f"Gemini Files API上传成功但缺少文件引用：{file_info}")
    try:
        gemini_wait_file(file_name_api)
        prompt = f"""
你是严谨的物流报价表数据提取专家。

当前供应商：{supplier}
当前上传文件名：{file_name}
目标国家：{target_country}
Python固定重量点：{json.dumps(FIXED_WEIGHTS, ensure_ascii=False)}

下面是唯一有效的供应商 Mapping 规则。请把其中的“Sheet定位类型、Sheet定位值、行定位类型、行定位值、列定位类型、列定位值、原始提取类型、Python解析器、Python规则参数、AI指令、说明”等全部视为规则说明，严格执行；不要用你自己的物流经验替代这些规则。

【Mapping】
{mapping_text}

【任务】
请完整阅读当前上传的Excel文件，并按照上述Mapping处理目标国家“{target_country}”。不要只检查一个Sheet；需要根据Mapping中每个字段的位置定位、内容、格式等要求，在整个文件范围内寻找每个字段的信息。
1. ID。请按照Mapping从Sheet名称等信息取得ID。
2. Destination Country。只输出实际开通目标国家“{target_country}”的线路；没有目标国家数据的线路不要输出。
3. Weight (kg)。对每条线路输出Python固定提供的全部Weight (kg)：{json.dumps(FIXED_WEIGHTS, ensure_ascii=False)}。不得修改重量、不得增加或删除重量。
4. RMB /kg和RMB /parcel。对每个Weight (kg)重量，按照Mapping在文件中寻找该重量实际对应的RMB /kg和RMB /parcel。如果文件中不存在该Weight (kg)对应的RMB /kg和RMB /parcel，必须返回RMB /kg=null、RMB /parcel=null；不得把其他重量段价格延伸过去，不得猜测。
5. Supplier固定为当前供应商“{supplier}”。
6. RMB in total、USD in total不要计算，交给Python/Google Sheets处理。
7. 无法确认的信息返回null；mapping有特殊、明确要求的，按mapping要求执行。
8. 严格按照Mapping中每个字段的位置定位、内容、格式等要求，在整个文件范围内寻找每个字段的信息，按Mapping规定的输出格式输出内容。

必须返回合法JSON，结构严格如下：
{{
  "routes": [
    {{
      "sheet": "线路Sheet名称",
      "ID": "...",
      "Destination Country": "{target_country}",
      "Supplier": "{supplier}",
      "Cargo Category": "...",
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
        {{"Weight (kg)": 0.25, "RMB /kg": null, "RMB /parcel": null}},
        {{"Weight (kg)": 0.50, "RMB /kg": null, "RMB /parcel": null}}
      ]
    }}
  ]
}}

注意：Weight Prices必须完整包含上述全部12个Python固定重量点。
"""
        body = {
            "contents": [{"parts": [{"text": prompt}, {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        r = requests.post(f"{GEMINI_BASE}/models/{AI_MODEL}:generateContent?key={AI_API_KEY}", json=body, timeout=300)
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise RuntimeError(f"Gemini API错误：{r.status_code} - {detail}")
        data = r.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if p.get("text"))
        if not text:
            raise RuntimeError(f"Gemini返回空结果：{data}")
        result = parse_json_response(text)
        if not isinstance(result.get("routes"), list):
            raise RuntimeError("AI返回格式错误：缺少routes数组")
        return result
    finally:
        gemini_delete_file(file_name_api)


# ============================================================
# ⑦ AI结果规范化：Python只做格式保护，不重新解释Mapping
# ============================================================
def normalize_route(route: Dict[str, Any], supplier: str, target_country: str) -> Dict[str, Any]:
    out = {
        "sheet": norm(route.get("sheet")),
        "ID": norm(route.get("ID")),
        "Destination Country": target_country,
        "Supplier": supplier,
        "Cargo Category": norm(route.get("Cargo Category")),
        "Cargo forbidden": route.get("Cargo forbidden"),
        "Time Min (day)": route.get("Time Min (day)"),
        "Time Max (day)": route.get("Time Max (day)"),
        "Time Type (workday/nature day)": route.get("Time Type (workday/nature day)"),
        "Volume Limit (cm)": route.get("Volume Limit (cm)"),
        "Volume to Weight Parameter": route.get("Volume to Weight Parameter"),
        "Pick&Packing/parcel": route.get("Pick&Packing/parcel") or "unknown",
        "DDP": route.get("DDP") or "unknown",
        "Extra Tax Required": route.get("Extra Tax Required") or "unknown",
        "Tax Policy": route.get("Tax Policy"),
        "Weight Prices": [],
        "errors": [],
    }
    raw_prices = route.get("Weight Prices")
    price_map = {}
    if isinstance(raw_prices, list):
        for item in raw_prices:
            if not isinstance(item, dict):
                continue
            w = safe_float(item.get("Weight (kg)"))
            if w is None:
                continue
            price_map[round(w, 2)] = {"Weight (kg)": round(w, 2), "RMB /kg": item.get("RMB /kg"), "RMB /parcel": item.get("RMB /parcel")}
    for w in FIXED_WEIGHTS:
        x = price_map.get(w, {"Weight (kg)": w, "RMB /kg": None, "RMB /parcel": None})
        x["Weight (kg)"] = w
        out["Weight Prices"].append(x)
    if not out["ID"]:
        out["errors"].append("AI未提取ID")
    return out


def build_preview_rows(routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for route in routes:
        for item in route["Weight Prices"]:
            rows.append({
                "ID": route["ID"],
                "Destination Country": route["Destination Country"],
                "Supplier": route["Supplier"],
                "Cargo Category": route["Cargo Category"],
                "Weight (kg)": item["Weight (kg)"],
                "RMB /kg": item.get("RMB /kg"),
                "RMB /parcel": item.get("RMB /parcel"),
            })
    return rows


# ============================================================
# ⑧ Google Sheets写入：保留原来的更新/新增逻辑，并动态按表头找列
# ============================================================
def col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def value_for_formula_col(pos: Dict[str, int], field: str, row_no: int) -> str:
    if field not in pos:
        raise RuntimeError(f"目标数据表缺少字段：{field}")
    return f"{col_letter(pos[field] + 1)}{row_no}"


def ensure_header(ws):
    raw = ws.get_all_values()
    numbered = [(idx, row) for idx, row in enumerate(raw, start=1) if any(norm(x) for x in row)]
    if not numbered:
        ws.append_row(STANDARD_FIELDS, value_input_option="RAW")
        return STANDARD_FIELDS, []
    first_no, first_row = numbered[0]
    first_names = [norm(x) for x in first_row]
    is_header = first_no == 1 and any(k in first_names for k in PRIMARY_KEYS)
    if not is_header:
        ws.insert_row(STANDARD_FIELDS, 1, value_input_option="RAW")
        return STANDARD_FIELDS, [(no + 1, row) for no, row in numbered]
    final_header = first_names
    missing = [f for f in STANDARD_FIELDS if f not in final_header]
    if missing:
        start = len(final_header) + 1
        ws.update(f"{col_letter(start)}1:{col_letter(start + len(missing) - 1)}1", [missing], value_input_option="RAW")
        final_header += missing
    return final_header, numbered[1:]


def build_record(route: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ID": route["ID"],
        "Destination Country": route["Destination Country"],
        "Supplier": route["Supplier"],
        "Cargo Category": norm(route.get("Cargo Category")),
        "Cargo forbidden": norm(route.get("Cargo forbidden")),
        "Time Min (day)": fmt_num(route.get("Time Min (day)")),
        "Time Max (day)": fmt_num(route.get("Time Max (day)")),
        "Time Type (workday/nature day)": norm(route.get("Time Type (workday/nature day)")),
        "Volume Limit (cm)": norm(route.get("Volume Limit (cm)")),
        "Volume to Weight Parameter": norm(route.get("Volume to Weight Parameter")),
        "Weight (kg)": fmt_num(item.get("Weight (kg)"), 2),
        "RMB /kg": fmt_num(item.get("RMB /kg"), 2),
        "RMB /parcel": fmt_num(item.get("RMB /parcel"), 2),
        "Pick&Packing/parcel": norm(route.get("Pick&Packing/parcel")),
        "RMB in total": None,
        "USD in total": None,
        "DDP": norm(route.get("DDP")),
        "Extra Tax Required": norm(route.get("Extra Tax Required")),
        "Tax Policy": norm(route.get("Tax Policy")),
    }


def write_results(country: str, routes: List[Dict[str, Any]]) -> Tuple[int, int]:
    ws = get_country_worksheet(country)
    final_header, data_rows = ensure_header(ws)
    pos = {name: i for i, name in enumerate(final_header)}
    for field in STANDARD_FIELDS:
        if field not in pos:
            raise RuntimeError(f"目标数据表缺少标准字段：{field}")
    ncols = len(final_header)

    # 现有数据按唯一键建立索引
    old = {}
    for no, row in data_rows:
        key = tuple(norm(row[pos[k]]) if pos[k] < len(row) else "" for k in PRIMARY_KEYS)
        old.setdefault(key, no)

    updates, appends = [], []
    std_cols = sorted(pos[f] for f in STANDARD_FIELDS)
    runs = []
    for c in std_cols:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])

    records = [build_record(route, item) for route in routes for item in route["Weight Prices"]]
    append_start = max([no for no, _ in data_rows], default=1) + 1

    for record in records:
        key = tuple(record[k] for k in PRIMARY_KEYS)
        row_no = old.get(key)
        if row_no is None:
            row_no = append_start + len(appends)
        row_values = [""] * ncols
        for f in STANDARD_FIELDS:
            if f in {"RMB in total", "USD in total"}:
                continue
            row_values[pos[f]] = record[f]
        weight_ref = value_for_formula_col(pos, "Weight (kg)", row_no)
        kg_ref = value_for_formula_col(pos, "RMB /kg", row_no)
        parcel_ref = value_for_formula_col(pos, "RMB /parcel", row_no)
        pick_ref = value_for_formula_col(pos, "Pick&Packing/parcel", row_no)
        total_ref = value_for_formula_col(pos, "RMB in total", row_no)
        row_values[pos["RMB in total"]] = f'=IF(OR({weight_ref}="",{kg_ref}=""),"",{weight_ref}*{kg_ref}+IFERROR(VALUE({parcel_ref}),0)+IFERROR(VALUE({pick_ref}),0))'
        row_values[pos["USD in total"]] = f'=IF({total_ref}="","",{total_ref}*GOOGLEFINANCE("CURRENCY:CNYUSD"))'

        if key in old:
            for a, b in runs:
                updates.append({"range": f"{col_letter(a + 1)}{row_no}:{col_letter(b + 1)}{row_no}", "values": [row_values[a:b + 1]]})
        else:
            old[key] = row_no
            appends.append(row_values)

    if updates:
        ws.batch_update(updates, value_input_option="USER_ENTERED")
    if appends:
        ws.append_rows(appends, value_input_option="USER_ENTERED")
    return len(updates), len(appends)


# ============================================================
# ⑨ 页面主流程
# ============================================================
uploaded = st.file_uploader("上传供应商报价表（xlsx）", type=["xlsx", "xls"])
target_country = st.text_input("目标国家（如：美国）", "").strip()

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    try:
        all_sheets = load_excel(file_bytes)
        supplier, evidence = detect_supplier(uploaded.name)
        mapping_text = load_mapping_text(supplier)
        st.success(f"识别供应商：{supplier}（{evidence}）")
        st.write(f"Mapping：{supplier} ｜ Excel Sheet：{len(all_sheets)}个")

        if not target_country:
            st.info("请输入目标国家后开始解析。")
        else:
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            with st.spinner("Gemini正在读取完整Excel并按照Mapping提取……"):
                result = ai_extract_full_excel(file_hash, file_bytes, uploaded.name, supplier, mapping_text, target_country)
            raw_routes = result.get("routes", [])
            routes = [normalize_route(r, supplier, target_country) for r in raw_routes if isinstance(r, dict)]
            ok_routes = [r for r in routes if r.get("ID")]
            st.write(f"成功解析 {len(ok_routes)} 条线路")

            for route in ok_routes:
                with st.expander(f"{route['ID']} | {route['sheet']} | {route['Destination Country']}", expanded=True):
                    st.json({k: v for k, v in route.items() if k not in {"errors"}}, expanded=False)
                    if route.get("errors"):
                        st.warning("；".join(route["errors"]))

            failed = [r for r in routes if not r.get("ID")]
            if failed:
                with st.expander(f"解析失败的线路（{len(failed)}个）"):
                    for route in failed:
                        st.write(f"{route.get('sheet', '')}: {'；'.join(route.get('errors', []))}")

            if ok_routes and st.button("写入目标数据表（Google Sheets）"):
                updates, appends = write_results(target_country, ok_routes)
                st.success(f"写入完成：更新{updates}行、新增{appends}行，共{updates + appends}行")

    except Exception as e:
        st.error(str(e))
