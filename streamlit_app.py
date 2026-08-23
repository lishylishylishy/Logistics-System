import json
import re
import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI

# =============================================================================
# 【公开配置区】这里只放非机密的常规配置 (机密信息已通过 st.secrets 读取)
# =============================================================================
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.7-plus"

# 1. 规则配置库 (Sheet A) 的 ID
RULE_SHEET_ID = "1GjrPj2bKQZFz_ls5Y6ViI2fL_ovWcayN6ri58tiJErU"

# 2. 数据结果库 (Sheet B) 的 ID
DATA_SHEET_ID = "1MPodFY2AO4dvSOjY6oTDL4o0LwkbUhPqUJCkFVuICo8"

# 联合主键定义
COMPOSITE_PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]

# =============================================================================
# Google Sheet 客户端与规则读取 (安全读取 Secrets)
# =============================================================================
def get_gspread_client():
    try:
        # 提取存储在 st.secrets 中的 Google credentials
        creds_dict = dict(st.secrets["gcp_service_account"])
    except KeyError:
        st.error("❌ 严重错误: 未在 Streamlit Secrets 中找到 [gcp_service_account] 配置！")
        st.stop()

    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        creds_dict, ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    )
    return gspread.authorize(creds)

def load_strict_rules(supplier_code: str) -> pd.DataFrame:
    """严格读取指定供应商的规则 Tab，不存在则直接抛出异常，绝不使用默认规则"""
    client = get_gspread_client()
    sh = client.open_by_key(RULE_SHEET_ID)
    try:
        ws = sh.worksheet(supplier_code)
    except gspread.exceptions.WorksheetNotFound:
        raise ValueError(f"❌ 规则表(Sheet A)中不存在名为【{supplier_code}】的Tab！")
    
    records = ws.get_all_records()
    if not records:
        raise ValueError(f"❌ 规则表【{supplier_code}】内容为空！")
    return pd.DataFrame(records)

# =============================================================================
# 辅助处理函数
# =============================================================================
def call_qwen_llm(prompt_text: str, text_context: str) -> str:
    """调用 Qwen 提取数据"""
    try:
        api_key = st.secrets["ALIYUN_API_KEY"]
    except KeyError:
        return "error: Missing ALIYUN_API_KEY in secrets"

    if not prompt_text: return "unknown"
    
    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严谨的数据提取接口，直接回答结果，无须多余解释。"},
                {"role": "user", "content": f"{prompt_text}\n\n待分析文本：\n{text_context}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"llm_error: {str(e)}"

def safe_float(val):
    if pd.isna(val): return 0.0
    nums = re.findall(r'[\d\.]+', str(val))
    try: return float(nums[0]) if nums else 0.0
    except: return 0.0

def generate_weight_steps(excel_w_min: float, excel_w_max: float) -> list:
    """按 0.25kg 递增生成阶梯重量"""
    standard_intervals = [
        (0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00),
        (1.00, 1.25), (1.25, 1.50), (1.50, 1.75), (1.75, 2.00),
        (2.00, 2.25), (2.25, 2.50), (2.50, 2.75), (2.75, 3.00)
    ]
    valid_steps = []
    for s_min, s_max in standard_intervals:
        if s_max <= excel_w_min or s_min >= excel_w_max: continue
        w_min, w_max = max(s_min, excel_w_min), min(s_max, excel_w_max)
        if excel_w_min >= 1 and w_min == s_min: w_min = 1.00
        elif excel_w_min > 1 and w_min == excel_w_min: w_min = round(excel_w_min + 0.01, 2)
        if excel_w_max <= 1 and w_max == s_max: w_max = 1.00
        elif excel_w_max < 1 and w_max == excel_w_max: w_max = round(excel_w_max - 0.01, 2)
        valid_steps.append((round(w_min, 2), round(w_max, 2)))
    return valid_steps

def parse_excel_weight_string(weight_str: str) -> tuple:
    s = str(weight_str).replace(' ', '').upper().replace('KG', '')
    m1 = re.search(r'([\d\.]+)<W[≤<=]([\d\.]+)', s)
    if m1: return float(m1.group(1)), float(m1.group(2))
    m2 = re.search(r'W[≤<=]([\d\.]+)', s)
    if m2: return 0.0, float(m2.group(1))
    return 0.0, 999.0

# =============================================================================
# 核心解析引擎 (严格基于 Sheet A 规则映射)
# =============================================================================
def parse_with_rules(uploaded_file, target_country: str, rules_df: pd.DataFrame) -> pd.DataFrame:
    # 建立映射字典
    rule_map = {}
    for _, r in rules_df.iterrows():
        field = str(r.get("说明", r.get("字段名称", r.get("Field", "")))).strip()
        instr = str(r.get("Python / LLM 机器指令 (转译)", "")).strip()
        loc = str(r.get("列名称-定位", "")).strip()
        rule_map[field] = {"instruction": instr, "loc_col": loc}

    excel_file = pd.ExcelFile(uploaded_file)
    all_rows = []

    # 提取需要的列名，不写死
    country_col_name = rule_map.get("Destination Country", {}).get("loc_col", "")
    weight_col_name = rule_map.get("Weight Range (min kg)", {}).get("loc_col", "") 
    freight_col_name = rule_map.get("RMB /kg", {}).get("loc_col", "")
    reg_col_name = rule_map.get("RMB /parcel", {}).get("loc_col", "")
    time_col_name = rule_map.get("Time (workday/nature day)", {}).get("loc_col", "")
    vol_limit_col_name = rule_map.get("Volume Limit (cm)", {}).get("loc_col", "")

    for sheet_name in excel_file.sheet_names:
        if any(kw in sheet_name for kw in ["目录", "对应表", "邮编", "禁运", "异形", "VAT"]): continue
        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
        if df_raw.empty: continue

        # 1. ID 与货物类别提取 (通过正则指令)
        id_instr = rule_map.get("ID", {}).get("instruction", r"r'\(([A-Z0-9]+)\)'")
        regex_match = re.search(r"r'([^']+)'", id_instr)
        pattern = regex_match.group(1) if regex_match else r'\(([A-Z0-9]+)\)'
        id_res = re.search(pattern, sheet_name)
        channel_id = id_res.group(1) if id_res else sheet_name.strip()
        
        cargo_category = "Regular" if "普货" in sheet_name else "Sensitive"

        # 2. 定位表头和底部说明区域
        header_idx = -1
        instruction_start_idx = -1
        
        for idx, row in df_raw.iterrows():
            row_str = "".join([str(v) for v in row.values if pd.notna(v)])
            if header_idx == -1 and country_col_name and country_col_name in row_str:
                header_idx = idx
            if instruction_start_idx == -1 and any(kw in row_str for kw in ["价格使用说明", "申报及税费", "计重规则"]):
                instruction_start_idx = idx

        if header_idx == -1: continue

        headers = [str(h).strip() for h in df_raw.iloc[header_idx].values]
        df_data = df_raw.iloc[header_idx + 1 : instruction_start_idx if instruction_start_idx != -1 else None].copy()
        df_data.columns = headers

        # 3. 严格匹配用户配置的列名
        country_col = next((c for c in headers if country_col_name and country_col_name in c), None)
        weight_col = next((c for c in headers if weight_col_name and weight_col_name in c), None)
        freight_col = next((c for c in headers if freight_col_name and freight_col_name in c), None)
        reg_col = next((c for c in headers if reg_col_name and reg_col_name in c), None)
        time_col = next((c for c in headers if time_col_name and time_col_name in c), None)
        vol_limit_col = next((c for c in headers if vol_limit_col_name and vol_limit_col_name in c), None)

        if not country_col or not weight_col: continue

        df_target = df_data[df_data[country_col].astype(str).str.strip() == target_country]
        if df_target.empty: continue

        # 4. 抓取完整的底部说明文本 (不截断，全部发给 AI)
        instruction_text = ""
        if instruction_start_idx != -1:
            raw_text = df_raw.iloc[instruction_start_idx:].values.flatten()
            instruction_text = "\n".join([str(v).strip() for v in raw_text if pd.notna(v) and str(v).strip()])

        # 5. 读取 AI 提取指令并发起请求
        cargo_fb_instr = rule_map.get("Cargo forbidden", {}).get("instruction", "")
        tax_instr = rule_map.get("Tax Policy", {}).get("instruction", "")
        pack_instr = rule_map.get("Pick&Packing/parcel", {}).get("instruction", "")
        vol_param_instr = rule_map.get("Volume to Weight parameter", {}).get("instruction", "")

        cargo_forbidden = call_qwen_llm(cargo_fb_instr, instruction_text) if cargo_fb_instr else "unknown"
        tax_policy = call_qwen_llm(tax_instr, instruction_text) if tax_instr else "unknown"
        pick_packing_parcel = safe_float(call_qwen_llm(pack_instr, instruction_text)) if pack_instr else 0.0

        # 材积系数优先正则，其次 AI
        vol_match = re.search(r'/\s*(\d{4})', instruction_text)
        vol_to_weight = int(vol_match.group(1)) if vol_match else (safe_float(call_qwen_llm(vol_param_instr, instruction_text)) if vol_param_instr else 8000)
        if vol_to_weight == 0: vol_to_weight = 8000

        # 6. 遍历目标国家的行
        time_instr = rule_map.get("Time (workday/nature day)", {}).get("instruction", "")
        vol_instr = rule_map.get("Volume Limit (cm)", {}).get("instruction", "")

        for _, row in df_target.iterrows():
            time_cell_text = str(row[time_col]) if time_col else ""
            vol_cell_text = str(row[vol_limit_col]) if vol_limit_col else ""
            
            time_formatted = call_qwen_llm(time_instr, time_cell_text) if time_cell_text and time_instr else "unknown"
            volume_limit = call_qwen_llm(vol_instr, vol_cell_text) if vol_cell_text and vol_instr else "unknown"

            src_w_min, src_w_max = parse_excel_weight_string(row[weight_col])
            weight_steps = generate_weight_steps(src_w_min, src_w_max)

            r_kg = safe_float(row[freight_col]) if freight_col else 0.0
            r_parcel = safe_float(row[reg_col]) if reg_col else 0.0

            for w_min, w_max in weight_steps:
                total_rmb = round(w_max * r_kg + r_parcel + pick_packing_parcel, 2)
                all_rows.append({
                    "ID": channel_id,
                    "Destination Country": target_country,
                    "Cargo Category": cargo_category,
                    "Cargo forbidden": cargo_forbidden,
                    "Time (workday/nature day)": time_formatted,
                    "Volume Limit (cm)": volume_limit,
                    "Volume to Weight parameter": vol_to_weight,
                    "Weight Range (min kg)": w_min,
                    "Weight Range (max kg)": w_max,
                    "RMB /kg": r_kg,
                    "RMB /parcel": r_parcel,
                    "Pick&Packing/parcel": pick_packing_parcel,
                    "RMB in total": total_rmb,
                    "Tax Policy": tax_policy
                })
                
    return pd.DataFrame(all_rows)

def upsert_to_google_sheet_b(new_df: pd.DataFrame, target_country: str) -> int:
    """写入 Google Sheet B 结果表"""
    client = get_gspread_client()
    sh = client.open_by_key(DATA_SHEET_ID)
    try:
        ws = sh.worksheet(target_country)
        existing_df = pd.DataFrame(ws.get_all_records())
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=target_country, rows="1000", cols="20")
        existing_df = pd.DataFrame()

    if existing_df.empty:
        final_df = new_df
    else:
        existing_df['_pk'] = existing_df[COMPOSITE_PRIMARY_KEYS].astype(str).agg('_'.join, axis=1)
        new_df['_pk'] = new_df[COMPOSITE_PRIMARY_KEYS].astype(str).agg('_'.join, axis=1)
        filtered = existing_df[~existing_df['_pk'].isin(new_df['_pk'])].copy()
        final_df = pd.concat([filtered, new_df], ignore_index=True).drop(columns=['_pk'], errors='ignore')

    final_df_clean = final_df.fillna("")
    ws.clear()
    ws.update(values=[final_df_clean.columns.tolist()] + final_df_clean.values.tolist(), range_name="A1")
    return len(final_df)

# =============================================================================
# Streamlit 界面
# =============================================================================
st.set_page_config(page_title="完全规则映射解析器", layout="wide")
st.title("📦 完全规则映射解析器 (安全集成版)")
st.divider()

with st.sidebar:
    st.header("1. 输入与上传")
    supplier_code = st.text_input("供应商代码 (对应Sheet A的规则Tab名)", value="4PX").strip()
    uploaded_file = st.file_uploader("上传报价单 (Excel)", type=["xlsx", "xls"])
    
    st.divider()
    st.header("2. 目标执行")
    target_country = st.text_input("🎯 指定目的国", value="墨西哥").strip()
    btn_start = st.button("🚀 严格读取规则并解析", type="primary", disabled=(not uploaded_file or not target_country))

if btn_start and uploaded_file:
    with st.spinner(f"正在严格读取规则表【{supplier_code}】..."):
        try:
            strict_rules_df = load_strict_rules(supplier_code)
            st.success(f"✅ 成功加载规则【{supplier_code}】，共 {len(strict_rules_df)} 条指令。")
            with st.expander("查看当前映射规则明细"):
                st.dataframe(strict_rules_df)
        except Exception as e:
            st.error(str(e))
            st.stop() 

    with st.spinner(f"正在按规则解析 Excel，提取【{target_country}】数据..."):
        parsed_df = parse_with_rules(uploaded_file, target_country, strict_rules_df)
        if parsed_df.empty:
            st.error(f"❌ 未解析到国家为【{target_country}】的数据，请检查国家名或 Excel 结构与映射配置。")
            st.stop()
            
        st.dataframe(parsed_df, use_container_width=True)

        with st.spinner("正在写入数据到 Google Sheet B..."):
            try:
                count = upsert_to_google_sheet_b(parsed_df, target_country)
                st.success(f"🎉 成功！【{target_country}】工作表当前共有 {count} 条数据。")
            except Exception as e:
                st.error(f"❌ 写入失败: {str(e)}")
