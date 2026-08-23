import json
import re
import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI

# =============================================================================
# 【公开配置区】
# =============================================================================
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.7-plus"

try:
    RULE_SHEET_ID = st.secrets["RULE_SHEET_ID"]
    DATA_SHEET_ID = st.secrets["DATA_SHEET_ID"]
except KeyError as e:
    st.error(f"❌ 严重错误: 未在 Streamlit Secrets 中找到 {e} 配置！")
    st.stop()

COMPOSITE_PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]

# =============================================================================
# Google Sheet 连接
# =============================================================================
@st.cache_resource
def get_gspread_client():
    try:
        gcp_json_str = st.secrets["gcp_json"]
        creds_dict = json.loads(gcp_json_str, strict=False)
    except Exception as e:
        st.error(f"❌ 严重错误: GCP JSON 解析失败 - {str(e)}")
        st.stop()

    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        creds_dict, ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    )
    return gspread.authorize(creds)

def load_strict_rules(supplier_code: str) -> pd.DataFrame:
    client = get_gspread_client()
    sh = client.open_by_key(RULE_SHEET_ID)
    try:
        ws = sh.worksheet(supplier_code)
        records = ws.get_all_records()
        return pd.DataFrame(records)
    except Exception as e:
        raise ValueError(f"❌ 读取规则表【{supplier_code}】失败: {str(e)}")

# =============================================================================
# 核心大模型调用 (加入强力缓存，相同文本不再重复请求，速度提升数十倍！)
# =============================================================================
@st.cache_data(show_spinner=False, max_entries=500)
def call_qwen_llm(prompt_text: str, text_context: str) -> str:
    if not prompt_text or not str(text_context).strip(): 
        return "unknown"
    
    try:
        api_key = st.secrets["API_KEY"]
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严谨的数据提取专家。请仅根据用户提供的文本提取信息，直接回答提取结果，严格按照要求的数据格式，不要有任何多余的废话、解释或开头。"},
                {"role": "user", "content": f"提取指令：{prompt_text}\n\n待分析文本内容：\n{text_context}"}
            ],
            temperature=0.1 # 降低随机性，保证提取更准确
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
# 一次性读取 Excel 减少 IO 操作
# =============================================================================
@st.cache_data(show_spinner=False)
def load_excel_all_sheets(file_bytes) -> dict:
    return pd.read_excel(file_bytes, sheet_name=None, header=None)

# =============================================================================
# 核心解析引擎 
# =============================================================================
def parse_with_rules(all_sheets: dict, target_country: str, rules_df: pd.DataFrame):
    rule_map = {}
    for _, r in rules_df.iterrows():
        field = str(r.get("说明", r.get("字段名称", r.get("Field", "")))).strip()
        rule_map[field] = {
            "instruction": str(r.get("Python / LLM 机器指令 (转译)", "")).strip(),
            "loc_col": str(r.get("列名称-定位", "")).strip()
        }

    all_rows = []
    
    # 提取常用定位列名
    country_col_name = rule_map.get("Destination Country", {}).get("loc_col", "Destination Country")
    weight_col_name = rule_map.get("Weight Range (min kg)", {}).get("loc_col", "重量段") 
    freight_col_name = rule_map.get("RMB /kg", {}).get("loc_col", "运费")
    reg_col_name = rule_map.get("RMB /parcel", {}).get("loc_col", "挂号费")
    time_col_name = rule_map.get("Time (workday/nature day)", {}).get("loc_col", "时效")
    vol_limit_col_name = rule_map.get("Volume Limit (cm)", {}).get("loc_col", "标准尺寸")

    # === 构建进度条 UI ===
    sheet_names = list(all_sheets.keys())
    total_sheets = len(sheet_names)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, sheet_name in enumerate(sheet_names):
        # 更新界面进度
        progress_bar.progress((i + 1) / total_sheets)
        status_text.markdown(f"**正在分析线路 [{i+1}/{total_sheets}]:** `{sheet_name}`")
        
        if any(kw in sheet_name for kw in ["目录", "对应表", "邮编", "禁运", "异形", "VAT"]): 
            continue

        df_raw = all_sheets[sheet_name]
        if df_raw.empty: continue

        # 提取 ID 逻辑：查找括号内字符
        id_match = re.search(r'\(([A-Z0-9]+)\)', sheet_name)
        channel_id = id_match.group(1) if id_match else sheet_name.strip()
        cargo_category = "Regular" if "普货" in sheet_name else "Sensitive"

        # 定位表头和说明文本
        header_idx, instruction_start_idx = -1, -1
        for idx, row in df_raw.iterrows():
            row_str = "".join([str(v) for v in row.values if pd.notna(v)])
            if header_idx == -1 and country_col_name in row_str:
                header_idx = idx
            if instruction_start_idx == -1 and any(kw in row_str for kw in ["价格使用说明", "申报及税费", "计重规则"]):
                instruction_start_idx = idx

        if header_idx == -1: continue

        headers = [str(h).strip() for h in df_raw.iloc[header_idx].values]
        df_data = df_raw.iloc[header_idx + 1 : instruction_start_idx if instruction_start_idx != -1 else None].copy()
        df_data.columns = headers

        # 在表头中精确找列
        country_col = next((c for c in headers if country_col_name in c), None)
        weight_col = next((c for c in headers if weight_col_name in c), None)
        if not country_col or not weight_col: continue

        # 过滤目标国家
        df_target = df_data[df_data[country_col].astype(str).str.strip() == target_country]
        if df_target.empty: continue

        # 合并底部说明文本用于大模型读取
        instruction_text = ""
        if instruction_start_idx != -1:
            raw_text = df_raw.iloc[instruction_start_idx:].values.flatten()
            instruction_text = "\n".join([str(v).strip() for v in raw_text if pd.notna(v) and str(v).strip()])

        # 获取规则指令
        cargo_fb_instr = rule_map.get("Cargo forbidden", {}).get("instruction", "")
        tax_instr = rule_map.get("Tax Policy", {}).get("instruction", "")
        pack_instr = rule_map.get("Pick&Packing/parcel", {}).get("instruction", "")
        vol_param_instr = rule_map.get("Volume to Weight parameter", {}).get("instruction", "")

        # 调用大模型 (因有缓存，如果各表下方文字相同，会秒回)
        cargo_forbidden = call_qwen_llm(cargo_fb_instr, instruction_text) if cargo_fb_instr else "unknown"
        tax_policy = call_qwen_llm(tax_instr, instruction_text) if tax_instr else "unknown"
        pick_packing_parcel = safe_float(call_qwen_llm(pack_instr, instruction_text)) if pack_instr else 0.0

        # 体积系数（优先正则抓取，失败再走LLM）
        vol_match = re.search(r'/\s*(\d{4})', instruction_text)
        vol_to_weight = int(vol_match.group(1)) if vol_match else (safe_float(call_qwen_llm(vol_param_instr, instruction_text)) if vol_param_instr else 8000)
        if vol_to_weight == 0: vol_to_weight = 8000

        # 行级时效与尺寸读取
        time_instr = rule_map.get("Time (workday/nature day)", {}).get("instruction", "")
        vol_instr = rule_map.get("Volume Limit (cm)", {}).get("instruction", "")
        time_col = next((c for c in headers if time_col_name in c), None)
        vol_limit_col = next((c for c in headers if vol_limit_col_name in c), None)
        freight_col = next((c for c in headers if freight_col_name in c), None)
        reg_col = next((c for c in headers if reg_col_name in c), None)

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
    
    # 扫尾 UI 处理
    status_text.markdown("✅ **分析完毕，正在整理数据...**")
    progress_bar.empty()
    return pd.DataFrame(all_rows)

def upsert_to_google_sheet_b(new_df: pd.DataFrame, target_country: str) -> int:
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
st.title("📦 完全规则映射解析器 (安全加速版)")
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
    # 步骤 1：读取规则
    with st.spinner(f"正在严格读取规则表【{supplier_code}】..."):
        try:
            strict_rules_df = load_strict_rules(supplier_code)
            st.success(f"✅ 成功加载规则【{supplier_code}】")
        except Exception as e:
            st.error(str(e))
            st.stop() 
            
    # 步骤 2：读取超级大的 Excel 到内存 (只需要做一次)
    file_bytes = uploaded_file.getvalue()
    with st.spinner(f"📦 正在将大 Excel 文件加载到内存 (9.7MB可能需要十几秒，请稍等)..."):
        all_sheets = load_excel_all_sheets(file_bytes)
        st.success(f"✅ Excel 解析完成，共发现 {len(all_sheets.keys())} 个工作表。")

    # 步骤 3：跑业务逻辑 + LLM
    st.markdown("### 🔄 正在通过 AI 提取数据")
    parsed_df = parse_with_rules(all_sheets, target_country, strict_rules_df)
    
    if parsed_df.empty:
        st.error(f"❌ 未在表格中成功提炼到国家为【{target_country}】的数据。")
        st.stop()
        
    st.dataframe(parsed_df, use_container_width=True)

    # 步骤 4：上传到谷歌表
    with st.spinner("正在写入最终数据到目标 Google Sheet ..."):
        try:
            count = upsert_to_google_sheet_b(parsed_df, target_country)
            st.success(f"🎉 成功！【{target_country}】工作表当前共有 {count} 条数据。")
        except Exception as e:
            st.error(f"❌ 写入失败: {str(e)}")
