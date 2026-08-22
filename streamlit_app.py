import json
import re
import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI

# =============================================================================
# 【配置区 A】核心接口与数据库配置
# =============================================================================
API_KEY = st.secrets.get("API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "kimi-k3"  # 推荐使用 qwen-plus

GOOGLE_SHEET_ID = "1MPodFY2AO4dvSOjY6oTDL4o0LwkbUhPqUJCkFVuICo8"
# 移除了固定的 TAB_NAME，因为现在将根据用户输入的国家名动态创建 Tab

# =============================================================================
# 【配置区 B】系统常量
# =============================================================================
SUPPLIER_CONFIGS = {
    "4PX": {"filename_keywords": ["递四方", "4px"], "default_volume_param": 8000},
    "YunExpress": {"filename_keywords": ["云途", "yunexpress"], "default_volume_param": 6000},
    "SF": {"filename_keywords": ["顺丰", "sf"], "default_volume_param": 6000}
}
COMPOSITE_PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]

# =============================================================================
# 工具与核心算法函数
# =============================================================================
def safe_float(val):
    if pd.isna(val): return 0.0
    nums = re.findall(r'[\d\.]+', str(val))
    try: return float(nums[0]) if nums else 0.0
    except: return 0.0

def detect_supplier(filename: str) -> str:
    fname_lower = filename.lower()
    for supplier_code, config in SUPPLIER_CONFIGS.items():
        if any(kw in fname_lower for kw in config["filename_keywords"]):
            return supplier_code
    return "Unknown"

@st.cache_data
def quick_scan_excel(uploaded_file):
    """【新增】快速扫描 Excel，不调用大模型，仅提取渠道名和国家列表供预览"""
    excel_file = pd.ExcelFile(uploaded_file)
    channels = []
    countries = set()
    
    for sheet in excel_file.sheet_names:
        if any(skip_kw in sheet for skip_kw in ["目录", "对应表", "邮编", "禁运", "异形", "VAT"]):
            continue
        
        channels.append(sheet)
        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, header=None, nrows=100) # 只扫前100行找表头
        if df_raw.empty: continue
        
        header_idx = -1
        for idx, row in df_raw.iloc[:15].iterrows():
            row_str = "".join([str(v) for v in row.values if pd.notna(v)])
            if "国家" in row_str or "目的国" in row_str:
                header_idx = idx
                break
        
        if header_idx != -1:
            headers = [str(h).replace('\n', ' ').strip() for h in df_raw.iloc[header_idx].values]
            country_col = next((i for i, c in enumerate(headers) if "国家" in c or "目的" in c), None)
            if country_col is not None:
                for val in df_raw.iloc[header_idx+1:, country_col].dropna().unique():
                    val_str = str(val).strip()
                    if val_str and val_str.lower() != 'nan' and not any(kw in val_str for kw in ["说明", "承诺", "注", "以上"]):
                        countries.add(val_str)
                        
    return channels, sorted(list(countries))

def parse_unstructured_text_with_llm(text_context: str, target_country: str) -> dict:
    """【修改】专门针对用户指定的目标国家提取规则"""
    default_res = {"Cargo_forbidden": "unknown", "Volume_Limit": "unknown", "Pick_Packing_parcel": "unknown", "Tax_Policy": "unknown"}
    if not API_KEY or API_KEY.startswith("sk-请填入"): return default_res
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        prompt = f"""
        请阅读以下物流报价表说明，专门针对【目的地国家：{target_country}】提取适用规则。
        （如果说明中没有单独提到该国，则提取全局通用规则）
        严格输出JSON：
        {{
            "Cargo_forbidden": "禁运物品列表，逗号分隔，无提及填 unknown",
            "Volume_Limit": "尺寸限制公式，如 55x40x25，无提及填 unknown",
            "Pick_Packing_parcel": "操作费或处理费(纯数字)，无提及填 0",
            "Tax_Policy": "清关模式或起征点，无提及填 unknown"
        }}
        说明文本：
        {text_context}
        """
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个物流数据提取助手，严格输出纯JSON。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"⚠️ AI提取【{target_country}】规则异常: {e}")
        return default_res

def generate_weight_steps(excel_w_min: float, excel_w_max: float) -> list:
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

def parse_supplier_excel(uploaded_file, supplier_code: str, target_country: str) -> pd.DataFrame:
    excel_file = pd.ExcelFile(uploaded_file)
    all_parsed_rows = []
    supp_cfg = SUPPLIER_CONFIGS.get(supplier_code, {})

    for sheet_name in excel_file.sheet_names:
        if any(skip_kw in sheet_name for skip_kw in ["目录", "对应表", "邮编", "禁运", "异形", "VAT"]):
            continue

        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
        if df_raw.empty: continue

        header_idx = -1
        for idx, row in df_raw.iloc[:15].iterrows():
            row_str = "".join([str(v) for v in row.values if pd.notna(v)])
            if "国家" in row_str or "目的国" in row_str:
                header_idx = idx
                break
        if header_idx == -1: continue

        headers = [str(h).replace('\n', ' ').strip() for h in df_raw.iloc[header_idx].values]
        df_data = df_raw.iloc[header_idx + 1:].copy()
        df_data.columns = headers

        country_col = next((c for c in headers if "国家" in c or "目的" in c), None)
        weight_col = next((c for c in headers if "重量" in c), None)
        freight_col = next((c for c in headers if "运费" in c or "单价" in c), None)
        reg_col = next((c for c in headers if "挂号" in c or "处理" in c or "件" in c), None)
        time_col = next((c for c in headers if "时效" in c), None)

        if not country_col or not weight_col: continue

        # 【核心逻辑】仅过滤出我们要求的目标国家，不存在则直接跳过该 Sheet，不浪费资源
        df_target = df_data[df_data[country_col].astype(str).str.strip() == target_country]
        if df_target.empty:
            continue
        
        # 既然该 Sheet 有我们要找的国家，拼装文本并请求 AI（每个包含该国家的 Sheet 只请求 1 次）
        raw_text_list = [str(v) for v in df_raw.iloc[:header_idx].values.flatten() if pd.notna(v)]
        raw_text_list += [str(v) for v in df_raw.iloc[header_idx+5:].values.flatten() if pd.notna(v)]
        text_context = " ".join([t for t in raw_text_list if len(t) > 5])[:2000]

        vol_match = re.search(r'/\s*(\d{4})', text_context)
        volume_to_weight = int(vol_match.group(1)) if vol_match else supp_cfg.get("default_volume_param", 8000)
        
        id_match = re.search(r'[\(（]([A-Za-z0-9]+)[\)）]', sheet_name)
        channel_id = id_match.group(1) if id_match else sheet_name.strip()
        cargo_category = "Regular" if "普货" in sheet_name else "Sensitive"

        # 调用 AI 提取该国家的特定/全局规则
        ai_info = parse_unstructured_text_with_llm(text_context, target_country)
        pick_pack = safe_float(ai_info.get("Pick_Packing_parcel", 0))

        # 遍历这个目标国家的所有价格行，严格绑定信息
        for _, row in df_target.iterrows():
            raw_time = str(row[time_col]).strip() if time_col and pd.notna(row[time_col]) else "10-15 工作日"
            time_match = re.search(r'(\d+[-~]\d+|\d+)\s*(工作日|自然日|天)', raw_time)
            time_formatted = f"{time_match.group(1)} {time_match.group(2)}" if time_match else raw_time

            src_w_min, src_w_max = parse_excel_weight_string(str(row[weight_col]))
            weight_steps = generate_weight_steps(src_w_min, src_w_max)

            r_kg = safe_float(row[freight_col]) if freight_col else 0.0
            r_parcel = safe_float(row[reg_col]) if reg_col else 0.0

            for w_min, w_max in weight_steps:
                total_rmb = round(w_max * r_kg + r_parcel + pick_pack, 2)
                all_parsed_rows.append({
                    "ID": channel_id,
                    "Destination Country": target_country,  # 严格绑定目标国家
                    "Cargo Category": cargo_category,
                    "Cargo forbidden": ai_info.get("Cargo_forbidden", "unknown"),
                    "Time (workday/nature day)": time_formatted,
                    "Volume Limit (cm)": ai_info.get("Volume_Limit", "unknown"),
                    "Volume to Weight parameter": volume_to_weight,
                    "Weight Range (min kg)": w_min,
                    "Weight Range (max kg)": w_max,
                    "RMB /kg": r_kg,
                    "RMB /parcel": r_parcel,
                    "Pick&Packing/parcel": ai_info.get("Pick_Packing_parcel", 0),
                    "RMB in total": total_rmb,
                    "Tax Policy": ai_info.get("Tax_Policy", "unknown")
                })
                
    return pd.DataFrame(all_parsed_rows)

def get_gspread_client():
    creds_dict = json.loads(st.secrets["gcp_json"], strict=False)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        creds_dict, ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    )
    return gspread.authorize(creds)

def fetch_existing_data(target_country: str):
    """拉取指定国家 Sheet 的数据"""
    try:
        client = get_gspread_client()
        ws = client.open_by_key(GOOGLE_SHEET_ID).worksheet(target_country)
        records = ws.get_all_records()
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def upsert_to_google_sheet(new_df: pd.DataFrame, target_country: str) -> int:
    """【修改】根据目标国家动态读写特定 Tab，利用复合主键更新"""
    client = get_gspread_client()
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    
    try:
        ws = sh.worksheet(target_country)
        existing_data = ws.get_all_records()
        existing_df = pd.DataFrame(existing_data) if existing_data else pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        # 如果没有该国家的 Tab，自动创建
        ws = sh.add_worksheet(title=target_country, rows="2000", cols="20")
        existing_df = pd.DataFrame()

    if existing_df.empty:
        final_df = new_df
    else:
        # 根据 ID、国家、重量限制 组合成唯一主键 _pk
        existing_df['_pk'] = existing_df[COMPOSITE_PRIMARY_KEYS].astype(str).agg('_'.join, axis=1)
        new_df['_pk'] = new_df[COMPOSITE_PRIMARY_KEYS].astype(str).agg('_'.join, axis=1)
        
        # 过滤掉旧数据中被新数据覆盖的行（留存不重复的旧数据）
        filtered_existing = existing_df[~existing_df['_pk'].isin(new_df['_pk'])].copy()
        
        # 把新旧数据合并
        final_df = pd.concat([filtered_existing, new_df], ignore_index=True)
        final_df.drop(columns=['_pk'], errors='ignore', inplace=True)

    final_df_clean = final_df.fillna("")
    ws.clear()
    ws.update([final_df_clean.columns.values.tolist()] + final_df_clean.values.tolist())
    return len(final_df)


# =============================================================================
# 界面 UI
# =============================================================================
st.set_page_config(page_title="📦 多供应商物流报价 AI 解析系统", layout="wide")
st.title("📦 物流报价单单点解析 (按国家)")

gsheet_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/edit"
st.markdown(f"🔗 **[点击前往云端数据库查看/管理分国 Sheet]({gsheet_url})**")
st.divider()

with st.sidebar:
    st.header("1. 上传文件")
    uploaded_file = st.file_uploader("上传报价单 (Excel)", type=["xlsx", "xls"])
    
    supplier_code = "Unknown"
    channels = []
    countries = []
    
    if uploaded_file:
        supplier_code = detect_supplier(uploaded_file.name)
        st.markdown(f"**识别供应商:** `{supplier_code}`")
        
        with st.spinner("正在快扫文件内容..."):
            channels, countries = quick_scan_excel(uploaded_file)
            
    st.divider()
    
    st.header("2. 指定任务")
    target_country = st.text_input("🎯 输入要单独跑的国家名", placeholder="例如: 墨西哥 或 Mexico").strip()
    btn_start = st.button("🚀 抓取该国数据并同步", type="primary", disabled=(not uploaded_file or not target_country))

# 主屏幕区域展示逻辑
if uploaded_file and not btn_start:
    st.subheader("👀 文件概览 (无需调用 AI)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**发现渠道数量 (Sheet):** {len(channels)} 个")
        with st.expander("点击查看表内的渠道列表"):
            st.write(channels)
    with col2:
        st.markdown(f"**总计包含目的国:** {len(countries)} 个")
        with st.expander("点击查看识别到的国家名称（核对用）"):
            st.write(countries)
            
    st.info("👈 请在左侧侧边栏输入你想单独跑的国家名称，避免资源浪费。")

elif btn_start and uploaded_file:
    with st.spinner(f"正在全表地毯式搜索【{target_country}】的数据，并由 AI 校验规则..."):
        parsed_df = parse_supplier_excel(uploaded_file, supplier_code, target_country)
        
        if parsed_df.empty:
            st.warning(f"🤷‍♂️ 文件中似乎未找到名为【{target_country}】的数据行，请检查是否有拼写错误。")
        else:
            st.success(f"✅ 抓取成功！共提取出 {len(parsed_df)} 条阶梯运费记录。")
            st.subheader("📊 解析结果预览")
            st.dataframe(parsed_df.head(50), use_container_width=True)

            with st.spinner(f"正在更新 Google Sheet 中的【{target_country}】工作表..."):
                try:
                    total_count = upsert_to_google_sheet(parsed_df, target_country)
                    st.balloons()
                    st.success(f"🎉 同步成功！云端【{target_country}】表中现存 {total_count} 条最新数据。")
                except Exception as e:
                    st.error(f"❌ 同步到云端失败，报错信息: {e}")
