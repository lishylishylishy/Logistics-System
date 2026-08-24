import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI


# ============================================================
# ① 项目配置：需要修改的内容全部集中在这里
# ============================================================
RULE_SHEET_ID = st.secrets["RULE_SHEET_ID"]      # Mapping规则 Google Spreadsheet
DATA_SHEET_ID = st.secrets["DATA_SHEET_ID"]      # 最终数据 Google Spreadsheet
AI_API_KEY = st.secrets["API_KEY"]               # AI API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.7-plus"

# 最终数据唯一键：三者相同=同一条记录，新数据替换旧数据
PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]

# Python固定的12个标准重量段
STANDARD_WEIGHTS = [(0, .25), (.25, .5), (.5, .75), (.75, 1), (1, 1.25), (1.25, 1.5),
                    (1.5, 1.75), (1.75, 2), (2, 2.25), (2.25, 2.5), (2.5, 2.75), (2.75, 3)]


# ============================================================
# ② Google Sheets
# ============================================================
@st.cache_resource
def get_gsheet_client():
    creds = json.loads(st.secrets["gcp_json"], strict=False)
    scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    return gspread.authorize(ServiceAccountCredentials.from_json_keyfile_dict(creds, scopes))


def load_sheet_records(spreadsheet_id, worksheet_name):
    sh = get_gsheet_client().open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet_name)
    return pd.DataFrame(ws.get_all_records()), ws


def load_supplier_config():
    df, _ = load_sheet_records(RULE_SHEET_ID, "Supplier_Config")
    required = ["Supplier Code", "Supplier Name", "Enabled", "Detection Type", "Detection Value", "Mapping Sheet"]
    missing = [x for x in required if x not in df.columns]
    if missing: raise ValueError(f"Supplier_Config 缺少列：{', '.join(missing)}")
    return df


def load_mapping(mapping_sheet):
    df, _ = load_sheet_records(RULE_SHEET_ID, mapping_sheet)
    required = ["字段", "是否AI读取", "提取粒度", "记录唯一键", "Sheet定位类型", "Sheet定位值",
                "行定位类型", "行定位值", "列定位类型", "列定位值", "原始提取类型", "Python解析器",
                "AI指令", "是否必填"]
    missing = [x for x in required if x not in df.columns]
    if missing: raise ValueError(f"Mapping【{mapping_sheet}】缺少列：{', '.join(missing)}")
    return df


# ============================================================
# ③ 供应商自动识别
# ============================================================
def match_rule(text, rule_type, rule_value):
    text, rule_value = str(text).strip(), str(rule_value).strip()
    if rule_type == "exact": return text == rule_value
    if rule_type == "contains": return rule_value in text
    if rule_type == "regex":
        return bool(re.search(rule_value, text, re.I))
    return False


def detect_supplier(workbook):
    config = load_supplier_config()
    enabled = config[config["Enabled"].astype(str).str.lower().isin(["true", "1", "yes", "是"])]
    matched = []
    for _, row in enabled.iterrows():
        rule_type, rule_value = str(row["Detection Type"]), str(row["Detection Value"])
        score = sum(match_rule(s, rule_type, rule_value) for s in workbook.keys())
        if score: matched.append((score, row))
    if not matched: raise ValueError("无法识别供应商，请检查 Supplier_Config。")
    matched.sort(key=lambda x: x[0], reverse=True)
    if len(matched) > 1 and matched[0][0] == matched[1][0]:
        raise ValueError(f"供应商识别冲突：{matched[0][1]['Supplier Code']} / {matched[1][1]['Supplier Code']}")
    row = matched[0][1]
    return str(row["Supplier Code"]).strip(), str(row["Supplier Name"]).strip(), str(row["Mapping Sheet"]).strip()


# ============================================================
# ④ Excel读取
# ============================================================
@st.cache_data(show_spinner=False)
def load_excel(file_bytes):
    import io
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)


def norm(v):
    if v is None or pd.isna(v): return ""
    return re.sub(r"\s+", " ", str(v).replace("\u3000", " ")).strip()


# ============================================================
# ⑤ Mapping定位引擎
# ============================================================
def get_rule(rules, field):
    rows = rules[rules["字段"].astype(str).str.strip() == field]
    if rows.empty: raise ValueError(f"Mapping没有字段：{field}")
    return rows.iloc[0].to_dict()


def locate_sheets(all_sheets, rule):
    typ, val = str(rule["Sheet定位类型"]), str(rule["Sheet定位值"])
    return [s for s in all_sheets if match_rule(s, typ, val)]


def find_cell(df, locator_type, locator_value):
    locator_value = norm(locator_value)
    for r in range(len(df)):
        for c in range(df.shape[1]):
            text = norm(df.iat[r, c])
            ok = (locator_type == "exact_header" and text == locator_value) or \
                 (locator_type == "contains_header" and locator_value in text) or \
                 (locator_type == "exact_text" and text == locator_value) or \
                 (locator_type == "contains_text" and locator_value in text)
            if ok: return r, c
    return None


def find_column(df, rule):
    result = find_cell(df, str(rule["列定位类型"]), str(rule["列定位值"]))
    if not result: raise ValueError(f"找不到列：{rule['列定位类型']} / {rule['列定位值']}")
    return result


def find_country_rows(df, country_rule, target_country):
    header = find_column(df, country_rule)
    header_row, country_col = header
    end_row = find_section_start(df)
    rows = []
    current_country = ""
    for r in range(header_row + 1, end_row):
        val = norm(df.iat[r, country_col])
        if val: current_country = val
        if current_country == target_country: rows.append(r)
    return header_row, country_col, rows


def find_section_start(df):
    anchors = ["价格使用说明", "计重规则", "申报及税费"]
    for r in range(len(df)):
        row_text = " ".join(norm(x) for x in df.iloc[r].tolist() if norm(x))
        if any(a in row_text for a in anchors): return r
    return len(df)


def extract_section(df, anchor):
    start = None
    for r in range(len(df)):
        row_text = " ".join(norm(x) for x in df.iloc[r].tolist() if norm(x))
        if anchor in row_text:
            start = r
            break
    if start is None: return ""
    anchors = ["价格使用说明", "计重规则", "申报及税费"]
    lines = []
    for r in range(start, len(df)):
        row_text = " | ".join(norm(x) for x in df.iloc[r].tolist() if norm(x))
        if not row_text: continue
        if r > start and any(a in row_text for a in anchors if a != anchor): break
        lines.append(f"Excel Row {r + 1}: {row_text}")
    return "\n".join(lines)


# ============================================================
# ⑥ Python基础解析
# ============================================================
def safe_float(v):
    if v is None or pd.isna(v): return None
    m = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(v))
    return float(m.group().replace(",", "")) if m else None


def extract_id(sheet_name):
    m = re.search(r"[\(（]([A-Za-z0-9]+)[\)）]", sheet_name)
    if not m: raise ValueError(f"无法从Sheet名称提取ID：{sheet_name}")
    return m.group(1)


def cargo_category(sheet_name):
    if "普货" in sheet_name: return "Regular"
    if any(x in sheet_name for x in ["带电", "特货", "敏感"]): return "Sensitive"
    return None


def parse_weight_range(text):
    text = norm(text).replace("KG", "").replace("kg", "").replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)<W[≤<=](\d+(?:\.\d+)?)", text)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.search(r"W[≤<=](\d+(?:\.\d+)?)", text)
    if m: return 0.0, float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~～]\s*(\d+(?:\.\d+)?)", text)
    if m: return float(m.group(1)), float(m.group(2))
    return None, None


def generate_weights(source_min, source_max):
    result = []
    if source_min is None or source_max is None: return result
    for smin, smax in STANDARD_WEIGHTS:
        if smax <= source_min or smin >= source_max: continue
        wmin, wmax = max(smin, source_min), min(smax, source_max)
        if source_min >= 1 and wmin == smin: wmin = 1
        elif source_min > 1 and wmin == source_min: wmin = round(source_min + .01, 2)
        if source_max <= 1 and wmax == smax: wmax = 1
        elif source_max < 1 and wmax == source_max: wmax = round(source_max - .01, 2)
        result.append((round(wmin, 2), round(wmax, 2)))
    return result


# ============================================================
# ⑦ AI
# ============================================================
@st.cache_data(show_spinner=False, max_entries=500)
def ai_json(prompt, context):
    client = OpenAI(api_key=AI_API_KEY, base_url=BASE_URL)
    r = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": "你是严谨的物流报价数据提取专家。只能使用输入内容，不得猜测。必须返回合法JSON。"},
            {"role": "user", "content": f"{prompt}\n\n原始数据：\n{context}"}
        ]
    )
    raw = r.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def ai_metadata(target_country, country_context, note_text, weight_text, tax_text, rules):
    fields = ["Cargo forbidden", "Time (workday/nature day)", "Volume Limit (cm)",
              "Volume to Weight parameter", "Pick&Packing/parcel", "Tax Policy"]
    instructions = []
    for field in fields:
        rule = get_rule(rules, field)
        if str(rule["是否AI读取"]).lower() in ["是", "true", "1", "yes"]:
            instructions.append(f"{field}: {norm(rule['AI指令'])}")
    prompt = f"""
目标国家：{target_country}

请按照以下要求提取信息：
{chr(10).join(instructions)}

严格返回：
{{
"Cargo forbidden": [],
"Time": {{"min": null, "max": null, "unit": null}},
"Volume Limit": {{"length_cm": null, "width_cm": null, "height_cm": null, "max_length_cm": null, "max_volume_m3": null, "formula": null, "raw": null}},
"Volume to Weight parameter": null,
"Pick&Packing/parcel": null,
"Tax Policy": {{"delivery_term": null, "fob_limit_usd": null, "cif_limit_usd": null, "raw": null}}
}}

只提取{target_country}，没有明确数据返回null，不得猜测。
"""
    context = f"目标国家价格行：\n{country_context}\n\n价格使用说明：\n{note_text}\n\n计重规则：\n{weight_text}\n\n申报及税费：\n{tax_text}"
    return ai_json(prompt, context)


def ai_weight_map(target_country, weight_rows, rule):
    prompt = f"""
目标国家：{target_country}

分析下面物流报价表的原始重量区间。

{norm(rule['AI指令'])}

你必须：
1. 判断source_min_kg和source_max_kg。
2. 标准目标max必须是0.25、0.50、0.75、1.00、1.25、1.50、1.75、2.00、2.25、2.50、2.75、3.00。
3. 每个target_max_kg必须选择真实存在的source_excel_row。
4. 选择能够覆盖该target_max_kg的源重量区间。
5. 超出源最大计费重量的target_max_kg返回null。
6. 不允许创造行号。

严格返回：
{{"source_min_kg": null, "source_max_kg": null, "mapping":[{{"target_max_kg":0.25,"source_excel_row":12}}]}}
"""
    return ai_json(prompt, json.dumps(weight_rows, ensure_ascii=False))


# ============================================================
# ⑧ AI结果格式化
# ============================================================
def format_time(v):
    if not v: return None
    if isinstance(v, dict):
        mn, mx, unit = v.get("min"), v.get("max"), v.get("unit")
        if mn is None: return None
        return f"{mn} {unit}" if mx is None or mn == mx else f"{mn}~{mx} {unit}"
    return norm(v)


def format_dimension(v):
    if not v: return None
    if not isinstance(v, dict): return norm(v)
    parts = []
    if v.get("length_cm") is not None and v.get("width_cm") is not None and v.get("height_cm") is not None:
        parts.append(f"{v['length_cm']}×{v['width_cm']}×{v['height_cm']} cm")
    if v.get("max_length_cm") is not None: parts.append(f"max_length={v['max_length_cm']}cm")
    if v.get("max_volume_m3") is not None: parts.append(f"max_volume={v['max_volume_m3']}m³")
    if v.get("formula"): parts.append(f"formula={v['formula']}")
    return "; ".join(parts) or v.get("raw")


def format_tax(v):
    if not v: return None
    if not isinstance(v, dict): return norm(v)
    parts = [v["delivery_term"]] if v.get("delivery_term") else []
    if v.get("fob_limit_usd") is not None: parts.append(f"FOB < {v['fob_limit_usd']} USD")
    if v.get("cif_limit_usd") is not None: parts.append(f"CIF < {v['cif_limit_usd']} USD")
    return ", ".join(parts) or v.get("raw")


def format_forbidden(v):
    if isinstance(v, list): return ", ".join(norm(x) for x in v if norm(x))
    return norm(v)


# ============================================================
# ⑨ 单个线路Sheet解析
# ============================================================
def parse_one_sheet(df, sheet_name, target_country, rules):
    rows, errors = [], []

    id_rule = get_rule(rules, "ID")
    country_rule = get_rule(rules, "Destination Country")
    weight_rule = get_rule(rules, "Weight Range (min kg)")
    freight_rule = get_rule(rules, "RMB /kg")
    parcel_rule = get_rule(rules, "RMB /parcel")

    channel_id = extract_id(sheet_name)
    cargo = cargo_category(sheet_name)

    header_row, country_col, country_rows = find_country_rows(df, country_rule, target_country)
    if not country_rows: return rows, errors

    weight_col = find_cell(df, weight_rule["列定位类型"], weight_rule["列定位值"])
    freight_col = find_cell(df, freight_rule["列定位类型"], freight_rule["列定位值"])
    parcel_col = find_cell(df, parcel_rule["列定位类型"], parcel_rule["列定位值"])
    if not weight_col or not freight_col or not parcel_col:
        return rows, [{"Sheet": sheet_name, "Field": "Price Columns", "Error": "无法定位重量/运费/挂号费列"}]

    weight_col_idx, freight_col_idx, parcel_col_idx = weight_col[1], freight_col[1], parcel_col[1]

    weight_source_rows = []
    country_context = []

    for r in country_rows:
        values = {f"Column_{c+1}": norm(df.iat[r, c]) for c in range(df.shape[1]) if norm(df.iat[r, c])}
        country_context.append({"Excel Row": r + 1, "Values": values})
        weight_raw = norm(df.iat[r, weight_col_idx])
        if weight_raw:
            weight_source_rows.append({
                "source_excel_row": r + 1,
                "weight_range_raw": weight_raw,
                "freight_raw": norm(df.iat[r, freight_col_idx]),
                "parcel_raw": norm(df.iat[r, parcel_col_idx])
            })

    metadata = ai_metadata(
        target_country,
        json.dumps(country_context, ensure_ascii=False),
        extract_section(df, "价格使用说明"),
        extract_section(df, "计重规则"),
        extract_section(df, "申报及税费"),
        rules
    )

    weight_ai = ai_weight_map(
        target_country,
        weight_source_rows,
        get_rule(rules, "Weight Range (max kg)")
    )

    source_min, source_max = safe_float(weight_ai.get("source_min_kg")), safe_float(weight_ai.get("source_max_kg"))
    if source_min is None or source_max is None:
        return rows, [{"Sheet": sheet_name, "Field": "Weight Range", "Error": "AI无法确定源重量范围"}]

    steps = generate_weights(source_min, source_max)
    mapping = {}
    for item in weight_ai.get("mapping", []):
        mx = safe_float(item.get("target_max_kg"))
        sr = item.get("source_excel_row")
        if mx is not None and sr is not None:
            mapping[round(mx, 2)] = int(sr)

    pick_pack = safe_float(metadata.get("Pick&Packing/parcel"))
    volume_param = safe_float(metadata.get("Volume to Weight parameter"))

    for wmin, wmax in steps:
        source_row = mapping.get(round(wmax, 2))
        if source_row is None:
            errors.append({"Sheet": sheet_name, "Field": "Weight Range", "Weight max": wmax, "Error": "AI没有指定对应源价格行"})
            continue

        r = source_row - 1
        if r < 0 or r >= len(df) or r not in country_rows:
            errors.append({"Sheet": sheet_name, "Field": "Weight Range", "Weight max": wmax, "Error": f"AI指定的Excel Row {source_row}不属于目标国家"})
            continue

        rkg = safe_float(df.iat[r, freight_col_idx])
        rparcel = safe_float(df.iat[r, parcel_col_idx])

        total = None if None in [rkg, rparcel, pick_pack] else round(wmax * rkg + rparcel + pick_pack, 2)

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
            "Tax Policy": format_tax(metadata.get("Tax Policy"))
        })

    return rows, errors


# ============================================================
# ⑩ 总解析：供应商已经自动识别，国家由App输入
# ============================================================
def parse_workbook(all_sheets, target_country, rules):
    id_rule = get_rule(rules, "ID")
    target_sheets = locate_sheets(all_sheets, id_rule)
    if not target_sheets: raise ValueError("没有找到符合当前供应商Mapping的线路Sheet。")

    all_rows, all_errors = [], []
    progress = st.progress(0)
    status = st.empty()

    for i, sheet_name in enumerate(target_sheets, 1):
        status.markdown(f"**正在解析 [{i}/{len(target_sheets)}]** `{sheet_name}` → `{target_country}`")
        try:
            rows, errors = parse_one_sheet(all_sheets[sheet_name], sheet_name, target_country, rules)
            all_rows.extend(rows)
            all_errors.extend(errors)
        except Exception as e:
            all_errors.append({"Sheet": sheet_name, "Field": "Parser", "Error": str(e)})
        progress.progress(i / len(target_sheets))

    progress.empty()
    status.success("✅ 解析完成")

    result = pd.DataFrame(all_rows)
    errors = pd.DataFrame(all_errors)

    if not result.empty:
        result = result.drop_duplicates(subset=PRIMARY_KEYS, keep="last").reset_index(drop=True)

    return result, errors


# ============================================================
# ⑪ 新旧数据比较
# ============================================================
def get_country_worksheet(country):
    sh = get_gsheet_client().open_by_key(DATA_SHEET_ID)
    try:
        return sh.worksheet(country)
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=country, rows="2000", cols="30")


def compare_data(new_df, old_df):
    if old_df.empty:
        return {"new": new_df, "updated": pd.DataFrame(), "unchanged": pd.DataFrame(), "final": new_df}

    new = new_df.copy()
    old = old_df.copy()

    for c in PRIMARY_KEYS:
        new[c] = new.get(c, "")
        old[c] = old.get(c, "")

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
        changed = any(norm(n.get(c, "")) != norm(o.get(c, "")) for c in new.columns if c != "_pk" and c in old.columns)
        (updated_rows if changed else unchanged_rows).append(n.drop("_pk").to_dict())

    untouched = old[~old["_pk"].isin(new["_pk"])].drop(columns="_pk", errors="ignore")
    final = pd.concat([untouched, pd.DataFrame(new_rows), pd.DataFrame(updated_rows), pd.DataFrame(unchanged_rows)], ignore_index=True)

    return {"new": pd.DataFrame(new_rows), "updated": pd.DataFrame(updated_rows),
            "unchanged": pd.DataFrame(unchanged_rows), "final": final}


def write_data(ws, df):
    df = df.fillna("")
    ws.clear()
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist(), range_name="A1")
    return len(df)


# ============================================================
# ⑫ App页面
# ============================================================
st.subheader("① 上传报价表")
uploaded_file = st.file_uploader("把供应商报价 Excel 拖到这里", type=["xlsx", "xls"])

st.subheader("② 输入目标国家/地区")
target_country = st.text_input("目标国家/地区", placeholder="例如：墨西哥、美国、加拿大").strip()

if uploaded_file:
    st.info(f"已选择文件：{uploaded_file.name}")

if st.button("🚀 识别供应商并开始解析", type="primary", use_container_width=True,
             disabled=not uploaded_file or not target_country):
    try:
        all_sheets = load_excel(uploaded_file.getvalue())
        supplier_code, supplier_name, mapping_sheet = detect_supplier(all_sheets)

        st.success(f"✅ 供应商：{supplier_name}（{supplier_code}）")
        st.info(f"✅ 自动加载 Mapping：{mapping_sheet}")

        rules = load_mapping(mapping_sheet)

        with st.spinner(f"正在解析目标国家：{target_country}"):
            parsed_df, errors_df = parse_workbook(all_sheets, target_country, rules)

        if parsed_df.empty:
            st.error(f"❌ 没有提取到【{target_country}】的数据。")
            if not errors_df.empty: st.dataframe(errors_df, use_container_width=True)
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

        st.download_button("⬇️ 下载解析结果 CSV", parsed_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{supplier_code}_{target_country}.csv", mime="text/csv")

        st.warning(f"确认后将更新 Google Sheet【{target_country}】；唯一键：ID + Destination Country + Weight Range (max kg)。")

        if st.button("✅ 确认并更新 Google Sheet", type="primary", use_container_width=True):
            count = write_data(ws, comparison["final"])
            st.success(f"🎉 更新完成，共 {count} 条记录。")

    except Exception as e:
        st.error(f"❌ 运行失败：{e}")
