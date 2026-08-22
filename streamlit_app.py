import json
import re
import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI

# =============================================================================
# 【配置区 A】核心接口与数据库配置 (请填入你的真实信息)
# =============================================================================
# 1. 阿里云百炼 API 配置
API_KEY = st.secrets.get("API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.8-27b"

# 2. Google Sheet 数据库配置
GOOGLE_SHEET_ID = "1MPodFY2AO4dvSOjY6oTDL4o0LwkbUhPqUJCkFVuICo8"
GOOGLE_SHEET_TAB_NAME = "data"  # 你的Google表格左下角那个标签页的名字，如果是中文版新建的通常是"工作表1"

# =============================================================================
# 【配置区 B】供应商拓展与系统常量
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
    """安全提取单元格中的数字，过滤掉不可见字符或非数字内容"""
    if pd.isna(val): return 0.0
    nums = re.findall(r'[\d\.]+', str(val))
    try:
        return float(nums[0]) if nums else 0.0
    except:
        return 0.0

def detect_supplier(filename: str) -> str:
    fname_lower = filename.lower()
    for supplier_code, config in SUPPLIER_CONFIGS.items():
        if any(kw in fname_lower for kw in config["filename_keywords"]):
            return supplier_code
    return "Unknown"

def parse_unstructured_text_with_llm(text_context: str, country: str) -> dict:
    default_res = {"Cargo_forbidden": "unknown", "Volume_Limit": "unknown", "Pick_Packing_parcel": "unknown", "Tax_Policy": "unknown"}
    if not API_KEY or API_KEY.startswith("sk-请填入"):
        return default_res
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        prompt = f"""
        请阅读以下物流说明文本，针对【目的地国家：{country}】提取4个字段。
        严格输出JSON，不要有Markdown符号：
        {{
            "Cargo_forbidden": "禁运物品列表，逗号分隔",
            "Volume_Limit": "尺寸限制公式，如 55x40x25",
            "Pick_Packing_parcel": "操作费数字，无提及填 unknown",
            "Tax_Policy": "清关模式或起征点"
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
        st.warning(f"⚠️ AI解析【{country}】发生异常: {e}")
        return default_res

def generate_weight_steps(excel_w_min: float, excel_w_max: float) -> list:
    """按 0.25kg 离散化切分阶梯重段"""
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

def parse_supplier_excel(uploaded_file, supplier_code: str) -> pd.DataFrame:
    excel_file = pd.ExcelFile(uploaded_file)
    all_parsed_rows = []
    supp_cfg = SUPPLIER_CONFIGS.get(supplier_code, {})

    for sheet_name in excel_file.sheet_names:
        if any(skip_kw in sheet_name for skip_kw in ["目录", "对应表", "邮编", "禁运", "异形", "VAT"]):
            continue

        # 修复：兼容中文和英文括号
        id_match = re.search(r'[\(（]([A-Za-z0-9]+)[\)）]', sheet_name)
        channel_id = id_match.group(1) if id_match else sheet_name.strip()
        cargo_category = "Regular" if "普货" in sheet_name else "Sensitive"

        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
        if df_raw.empty: continue

        header_idx = -1
        for idx, row in df_raw.iloc[:15].iterrows():
            row_str = "".join([str(v) for v in row.values if pd.notna(v)])
            if "国家" in row_str or "目的国" in row_str:
                header_idx = idx
                break
        if header_idx == -1: continue

        text_context = " ".join([str(v) for v in df_raw.iloc[:header_idx].values.flatten() if pd.notna(v)])
        text_context += " " + " ".join([str(v) for v in df_raw.iloc[header_idx+10:].values.flatten() if pd.notna(v)])
        vol_match = re.search(r'/\s*(\d{4})', text_context)
        volume_to_weight = int(vol_match.group(1)) if vol_match else supp_cfg.get("default_volume_param", 8000)

        headers = [str(h).replace('\n', ' ').strip() for h in df_raw.iloc[header_idx].values]
        df_data = df_raw.iloc[header_idx + 1:].copy()
        df_data.columns = headers

        country_col = next((c for c in headers if "国家" in c or "目的" in c), None)
        weight_col = next((c for c in headers if "重量" in c), None)
        freight_col = next((c for c in headers if "运费" in c or "单价" in c), None)
        reg_col = next((c for c in headers if "挂号" in c or "处理" in c or "件" in c), None)
        time_col = next((c for c in headers if "时效" in c), None)

        if not country_col or not weight_col: continue
        ai_cache = {}

        for _, row in df_data.iterrows():
            country = str(row[country_col]).strip()
            
            # 修复：强力拦截底部备注说明等脏数据
            if not country or country.lower() == "nan" or any(kw in country for kw in ["说明", "承诺", "注", "以上", "计费", ":", "："]):
                continue

            raw_time = str(row[time_col]).strip() if time_col and pd.notna(row[time_col]) else "10-15 工作日"
            time_match = re.search(r'(\d+[-~]\d+|\d+)\s*(工作日|自然日|天)', raw_time)
            time_formatted = f"{time_match.group(1)} {time_match.group(2)}" if time_match else raw_time

            if country not in ai_cache:
                ai_cache[country] = parse_unstructured_text_with_llm(text_context[:3000], country)
            ai_info = ai_cache[country]

            src_w_min, src_w_max = parse_excel_weight_string(str(row[weight_col]))
            weight_steps = generate_weight_steps(src_w_min, src_w_max)

            # 修复：安全提取数字
            r_kg = safe_float(row[freight_col]) if freight_col else 0.0
            r_parcel = safe_float(row[reg_col]) if reg_col else 0.0
            pick_pack = safe_float(ai_info.get("Pick_Packing_parcel", 0))

            for w_min, w_max in weight_steps:
                total_rmb = round(w_max * r_kg + r_parcel + pick_pack, 2)
                all_parsed_rows.append({
                    "ID": channel_id,
                    "Destination Country": country,
                    "Cargo Category": cargo_category,
                    "Cargo forbidden": ai_info.get("Cargo_forbidden", "unknown"),
                    "Time (workday/nature day)": time_formatted,
                    "Volume Limit (cm)": ai_info.get("Volume_Limit", "unknown"),
                    "Volume to Weight parameter": volume_to_weight,
                    "Weight Range (min kg)": w_min,
                    "Weight Range (max kg)": w_max,
                    "RMB /kg": r_kg,
                    "RMB /parcel": r_parcel,
                    "Pick&Packing/parcel": ai_info.get("Pick_Packing_parcel", "unknown"),
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

def fetch_existing_data():
    """拉取云端现有数据用于展示"""
    try:
        client = get_gspread_client()
        ws = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_TAB_NAME)
        records = ws.get_all_records()
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def upsert_to_google_sheet(new_df: pd.DataFrame) -> int:
    """写入及合并数据到云端"""
    client = get_gspread_client()
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws = sh.worksheet(GOOGLE_SHEET_TAB_NAME)
        existing_data = ws.get_all_records()
        existing_df = pd.DataFrame(existing_data) if existing_data else pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=GOOGLE_SHEET_TAB_NAME, rows="2000", cols="20")
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
    ws.update([final_df_clean.columns.values.tolist()] + final_df_clean.values.tolist())
    return len(final_df)


# =============================================================================
# 界面 UI - 严格保持侧边栏与主区域分离的布局
# =============================================================================
st.set_page_config(page_title="📦 多供应商物流报价 AI 解析系统", layout="wide")
st.title("📦 多供应商物流报价单 AI 解析系统")

# 主区域顶部：提供云端数据库链接
gsheet_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/edit"
st.markdown(f"🔗 **存储总表链接：[点击此处直达 Google Sheet 查看/编辑云端数据]({gsheet_url})**")
st.divider()

# 左侧侧边栏：操作控制面板
with st.sidebar:
    st.header("控制面板")
    uploaded_file = st.file_uploader("上传报价单 (Excel)", type=["xlsx", "xls"])
    
    supplier_code = "Unknown"
    if uploaded_file:
        supplier_code = detect_supplier(uploaded_file.name)
        st.markdown(f"**识别供应商:** `{supplier_code}`")
        st.markdown(f"**当前调用大模型:** `{MODEL_NAME}`")
    
    btn_start = st.button("🚀 开始解析并同步至云端", type="primary", disabled=(not uploaded_file or supplier_code == "Unknown"))

# 中间主区域展示逻辑
if not uploaded_file or not btn_start:
    st.subheader("👀 云端数据库当前内容预览")
    with st.spinner("正在拉取 Google Sheet 最新数据..."):
        existing_df = fetch_existing_data()
        if existing_df.empty:
            st.info("当前云端表格为空，请在左侧上传文件开始解析。")
        else:
            st.dataframe(existing_df, use_container_width=True, height=500)
else:
    # 运行解析与写入
    with st.spinner(f"正在对【{supplier_code}】报价单进行解析，并调用 AI 提取复杂规则... (这可能需要一两分钟)"):
        parsed_df = parse_supplier_excel(uploaded_file, supplier_code)
        
        if parsed_df.empty:
            st.error("❌ 未能解析出有效数据，请检查文件格式。")
        else:
            st.success(f"✅ 解析成功！共提取出 {len(parsed_df)} 条有效记录。")
            st.subheader("📊 本次解析结果预览")
            st.dataframe(parsed_df.head(50), use_container_width=True)

            with st.spinner("正在执行复合主键比对，并将数据同步写入 Google Sheet..."):
                try:
                    total_count = upsert_to_google_sheet(parsed_df)
                    st.balloons()
                    st.success(f"🎉 同步成功！Google Sheet 云端现存 {total_count} 条最新有效数据。")
                except Exception as e:
                    st.error(f"❌ 同步到云端失败，报错信息: {e}")
