import json
import re
import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI

# =============================================================================
# 【配置区】Google Sheet ID 与 API 配置
# =============================================================================
API_KEY = st.secrets.get("API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"  # 阿里云百炼 Qwen 模型

# 1. 映射规则库 Google Sheet ID (读取提取规则)
RULE_SHEET_ID = st.secrets.get("RULE_SHEET_ID", "1MPodFY2AO4dvSOjY6oTDL4o0LwkbUhPqUJCkFVuICo8") 

# 2. 目标数据存储 Google Sheet ID (按国家写入 Tab)
DATA_SHEET_ID = st.secrets.get("DATA_SHEET_ID", "1MPodFY2AO4dvSOjY6oTDL4o0LwkbUhPqUJCkFVuICo8")

COMPOSITE_PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]

# =============================================================================
# 核心组件：Google Sheet 客户端与规则读取 Engine
# =============================================================================
def get_gspread_client():
    creds_dict = json.loads(st.secrets["gcp_json"], strict=False)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        creds_dict, ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    )
    return gspread.authorize(creds)

def safe_float(val):
    if pd.isna(val): return 0.0
    nums = re.findall(r'[\d\.]+', str(val))
    try: return float(nums[0]) if nums else 0.0
    except: return 0.0

@st.cache_data(ttl=300)
def fetch_mapping_rules(supplier_code: str) -> dict:
    """【核心】从规则配置 Google Sheet A 读取该供应商的提取指令字典"""
    try:
        client = get_gspread_client()
        sh = client.open_by_key(RULE_SHEET_ID)
        # 尝试寻找该供应商命名的 Rule Tab，没有则读取默认 'default'
        try:
            ws = sh.worksheet(supplier_code)
        except:
            ws = sh.worksheet("default")
            
        records = ws.get_all_records()
        rules = {}
        for r in records:
            field_name = str(r.get("字段名称", r.get("Field", ""))).strip()
            instruction = str(r.get("Python / LLM 机器指令 (转译)", r.get("Instruction", ""))).strip()
            loc_name = str(r.get("行名称-定位列名称-定位", r.get("Location", ""))).strip()
            if field_name:
                rules[field_name] = {"instruction": instruction, "location": loc_name}
        return rules
    except Exception as e:
        st.warning(f"⚠️ 读取规则配置表失败，采用系统预设规则: {e}")
        return {}

# =============================================================================
# 规则转译引擎 (Rule Processors)
# =============================================================================
def call_llm_prompt(prompt_text: str, text_context: str) -> str:
    """【API 通用防护】调用 Qwen 模型，不传易引发 400 错误的额外参数"""
    if not API_KEY: return "unknown"
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        full_prompt = f"{prompt_text}\n\n待分析说明文本：\n{text_context}"
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严谨的物流数据提取接口，直接回答结果，无须多余解释。"},
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "unknown"

def generate_weight_steps(excel_w_min: float, excel_w_max: float) -> list:
    """【PYTHON_STEPPER】0.25kg 阶梯递增与边界修正引擎"""
    standard_intervals = [
        (0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00),
        (1.00, 1.25), (1.25, 1.50), (1.50, 1.75), (1.75, 2.00),
        (2.00, 2.25), (2.25, 2.50), (2.50, 2.75), (2.75, 3.00)
    ]
    valid_steps = []
    for step_min, step_max in standard_intervals:
        if step_max <= excel_w_min or step_min >= excel_w_max: continue
        w_min, w_max = max(step_min, excel_w_min), min(step_max, excel_w_max)
        if excel_w_min >= 1 and w_min == step_min: w_min = 1.00
        elif excel_w_min > 1 and w_min == excel_w_min: w_min = round(excel_w_min + 0.01, 2)
        if excel_w_max <= 1 and w_max == step_max: w_max = 1.00
        elif excel_w_max < 1 and w_max == excel_w_max: w_max = round(excel_w_max - 0.01, 2)
        valid_steps.append((round(w_min, 2), round(w_max, 2)))
    return valid_steps

def parse_excel_weight_string(weight_str: str) -> tuple:
    if not isinstance(weight_str, str): return 0.0, 999.0
    s = weight_str.replace(' ', '').replace('kg', '').replace('KG', '')
    m = re.search(r'([\d\.]+)\s*<\s*W\s*[≤<=]\s*([\d\.]+)', s)
    if m: return float(m.group(1)), float(m.group(2))
    m2 = re.search(r'W\s*[≤<=]\s*([\d\.]+)', s)
    if m2: return 0.0, float(m2.group(1))
    return 0.0, 999.0

# =============================================================================
# 执行引擎：结合 Excel 与 映射规则 进行解析
# =============================================================================
def parse_supplier_excel_with_rules(uploaded_file, target_country: str, rules: dict) -> pd.DataFrame:
    excel_file = pd.ExcelFile(uploaded_file)
    all_parsed_rows = []

    for sheet_name in excel_file.sheet_names:
        if any(skip_kw in sheet_name for skip_kw in ["目录", "对应表", "邮编", "禁运", "异形", "VAT"]):
            continue

        # 1. 提取 ID (REGEX_EXTRACT)
        id_rule = rules.get("ID", {}).get("instruction", r"REGEX_EXTRACT: r'\(([A-Z0-9]+)\)'")
        regex_pattern = re.search(r"r'([^']+)'", id_rule)
        pattern = regex_pattern.group(1) if regex_pattern else r'\(([A-Z0-9]+)\)'
        id_match = re.search(pattern, sheet_name)
        channel_id = id_match.group(1) if id_match else sheet_name.strip()

        # 2. 提取 Cargo Category
        cargo_category = "Regular" if "普货" in sheet_name else "Sensitive"

        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
        if df_raw.empty: continue

        # 查找表头位置
        header_idx = -1
        for idx, row in df_raw.iloc[:20].iterrows():
            row_str = "".join([str(v) for v in row.values if pd.notna(v)])
            if "国家" in row_str or "Destination Country" in row_str:
                header_idx = idx
                break
        if header_idx == -1: continue

        headers = [str(h).strip() for h in df_raw.iloc[header_idx].values]
        df_data = df_raw.iloc[header_idx + 1:].copy()
        df_data.columns = headers

        # 列定位 lookup
        country_col = next((c for c in headers if "国家" in c or "Destination" in c), None)
        weight_col = next((c for c in headers if "重量" in c or "Weight" in c), None)
        freight_col = next((c for c in headers if "运费" in c and "RMB/KG" in c), None) or next((c for c in headers if "运费" in c or "RMB/kg" in c), None)
        reg_col = next((c for c in headers if "挂号" in c and "RMB/票" in c), None) or next((c for c in headers if "挂号" in c or "RMB/parcel" in c), None)
        time_col = next((c for c in headers if "时效" in c or "Time" in c), None)

        if not country_col or not weight_col: continue

        # 过滤目标国家
        df_target = df_data[df_data[country_col].astype(str).str.strip() == target_country]
        if df_target.empty: continue

        # 上下文提取
        raw_text_list = [str(v) for v in df_raw.iloc[:header_idx].values.flatten() if pd.notna(v)]
        raw_text_list += [str(v) for v in df_raw.iloc[header_idx+5:].values.flatten() if pd.notna(v)]
        text_context = " ".join([t for t in raw_text_list if len(t) > 5])[:2000]

        # 材积系数
        vol_match = re.search(r'/\s*(\d{4})', text_context)
        volume_to_weight = int(vol_match.group(1)) if vol_match else 8000

        # LLM 动态提词解析
        forbidden_prompt = f"仅提取针对{target_country}的禁运物品规定，直接输出英文逗号分隔项，无提及输出 unknown"
        cargo_forbidden = call_llm_prompt(forbidden_prompt, text_context)

        volume_prompt = f"提取{target_country}在标准尺寸列的规定，直接提取尺寸限制与公式，无提及输出 unknown"
        volume_limit = call_llm_prompt(volume_prompt, text_context)

        pack_prompt = f"提取针对{target_country}的分拣打包杂费数字，只输出纯数字，无提及输出 0"
        pick_pack_str = call_llm_prompt(pack_prompt, text_context)
        pick_packing_parcel = safe_float(pick_pack_str)

        tax_prompt = f"提取{target_country}的清关模式(DDP/DDU)及FOB/CIF要求，转译为公式化表达，无提及输出 unknown"
        tax_policy = call_llm_prompt(tax_prompt, text_context)

        # 遍历生成阶梯数据
        for _, row in df_target.iterrows():
            raw_time = str(row[time_col]).strip() if time_col and pd.notna(row[time_col]) else "10-15 workday"
            time_match = re.search(r'(\d+[-~]\d+|\d+)\s*(工作日|自然日|天|workday)?', raw_time)
            time_formatted = raw_time if not time_match else f"{time_match.group(1)} {time_match.group(2) if time_match.group(2) else 'workday'}"

            src_w_min, src_w_max = parse_excel_weight_string(str(row[weight_col]))
            weight_steps = generate_weight_steps(src_w_min, src_w_max)

            r_kg = safe_float(row[freight_col]) if freight_col else 0.0
            r_parcel = safe_float(row[reg_col]) if reg_col else 0.0

            for w_min, w_max in weight_steps:
                total_rmb = round(w_max * r_kg + r_parcel + pick_packing_parcel, 2)
                
                all_parsed_rows.append({
                    "ID": channel_id,
                    "Destination Country": target_country,
                    "Cargo Category": cargo_category,
                    "Cargo forbidden": cargo_forbidden,
                    "Time (workday/nature day)": time_formatted,
                    "Volume Limit (cm)": volume_limit,
                    "Volume to Weight parameter": volume_to_weight,
                    "Weight Range (min kg)": w_min,
                    "Weight Range (max kg)": w_max,
                    "RMB /kg": r_kg,
                    "RMB /parcel": r_parcel,
                    "Pick&Packing/parcel": pick_packing_parcel,
                    "RMB in total": total_rmb,
                    "Tax Policy": tax_policy
                })
                
    return pd.DataFrame(all_parsed_rows)

def upsert_to_google_sheet(new_df: pd.DataFrame, target_country: str) -> int:
    """写入 Google Sheet B (分国家 Tab 覆盖写)"""
    client = get_gspread_client()
    sh = client.open_by_key(DATA_SHEET_ID)
    
    try:
        ws = sh.worksheet(target_country)
        existing_data = ws.get_all_records()
        existing_df = pd.DataFrame(existing_data) if existing_data else pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=target_country, rows="2000", cols="20")
        existing_df = pd.DataFrame()

    if existing_df.empty:
        final_df = new_df
    else:
        existing_df['_pk'] = existing_df[COMPOSITE_PRIMARY_KEYS].astype(str).agg('_'.join, axis=1)
        new_df['_pk'] = new_df[COMPOSITE_PRIMARY_KEYS].astype(str).agg('_'.join, axis=1)
        filtered_existing = existing_df[~existing_df['_pk'].isin(new_df['_pk'])].copy()
        final_df = pd.concat([filtered_existing, new_df], ignore_index=True)
        final_df.drop(columns=['_pk'], errors='ignore', inplace=True)

    final_df_clean = final_df.fillna("")
    ws.clear()
    data_matrix = [final_df_clean.columns.values.tolist()] + final_df_clean.values.tolist()
    try:
        ws.update(values=data_matrix, range_name="A1")
    except TypeError:
        ws.update(data_matrix)
        
    return len(final_df)

# =============================================================================
# Streamlit UI
# =============================================================================
st.set_page_config(page_title="规则映射驱动的物流报价解析器", layout="wide")
st.title("📦 规则映射驱动的物流报价解析器")

st.markdown(f"🔗 **[配置规则库 Google Sheet A]**(https://docs.google.com/spreadsheets/d/{RULE_SHEET_ID}) | 🔗 **[数据结果库 Google Sheet B]**(https://docs.google.com/spreadsheets/d/{DATA_SHEET_ID})")
st.divider()

with st.sidebar:
    st.header("1. 配置与上传")
    supplier_code = st.text_input("供应商代码 (对应规则Tab名)", value="4PX").strip()
    uploaded_file = st.file_uploader("上传报价单 (Excel)", type=["xlsx", "xls"])
    
    st.divider()
    st.header("2. 目标执行")
    target_country = st.text_input("🎯 指定目的国", value="墨西哥").strip()
    btn_start = st.button("🚀 加载映射规则并解析写入", type="primary", disabled=(not uploaded_file or not target_country))

if btn_start and uploaded_file:
    with st.spinner(f"1/3 正在从 Google Sheet A 读取【{supplier_code}】的映射规则..."):
        mapping_rules = fetch_mapping_rules(supplier_code)
        
    with st.spinner(f"2/3 正在根据规则提取【{target_country}】的数据并调用 Qwen 解析..."):
        parsed_df = parse_supplier_excel_with_rules(uploaded_file, target_country, mapping_rules)
        
        if parsed_df.empty:
            st.error(f"❌ 未抓取到【{target_country}】的数据，请检查 Excel 中国家列命名或输入国家是否正确。")
        else:
            st.success(f"✅ 解析成功！提取出 {len(parsed_df)} 条精确记录。")
            st.dataframe(parsed_df, use_container_width=True)

            with st.spinner(f"3/3 正在更新写回 Google Sheet B 中的【{target_country}】Tab..."):
                try:
                    total_count = upsert_to_google_sheet(parsed_df, target_country)
                    st.balloons()
                    st.success(f"🎉 同步完成！【{target_country}】工作表当前共 {total_count} 条数据。")
                except Exception as e:
                    st.error(f"❌ 写入失败: {e}")
