import json
import re
import pandas as pd
import streamlit as st
import gspread
from google import genai
from google.genai import types

# =============================================================================
# 【配置区 A】个人与 AI 模型配置 (初次部署 / 换密钥 / 换 AI 模型时修改)
# =============================================================================
# 1. Google Sheet 目标文档 ID
SPREADSHEET_ID = "1GjrPj2bKQZFz_ls5Y6ViI2fL_ovWcayN6ri58tiJErU"

# 2. Gemini API 密钥
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# 3. Gemini AI 模型版本 (API 与 模型名称放在一起，方便一键切换如 gemini-2.5-flash / gemini-2.0-flash)
GEMINI_MODEL = "gemini-2.5-flash"


# =============================================================================
# 【配置区 B】供应商匹配配置 (后期新增供应商在此添加关键词即可)
# =============================================================================
SUPPLIER_CONFIGS = {
    "4PX": {
        "filename_keywords": ["递四方", "4px"],     # 匹配 Excel 文件名里的关键词
        "default_volume_param": 8000              # 默认材积系数兜底值
    },
    "YunExpress": {
        "filename_keywords": ["云途", "yunexpress"],
        "default_volume_param": 6000
    },
    "SF": {
        "filename_keywords": ["顺丰", "sf"],
        "default_volume_param": 6000
    }
}


# =============================================================================
# 【配置区 C】系统数据库常量 (无需变动)
# =============================================================================
MASTER_DB_TAB_NAME = "Master_Price_DB"  # 汇总写入的云端页签名称
COMPOSITE_PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]  # 覆盖排重复合主键


# =============================================================================
# 1. 核心 AI 模块 (调用配置区 A 的 API Key 和 模型名称)
# =============================================================================
def parse_unstructured_text_with_gemini(api_key: str, model_name: str, text_context: str, country: str) -> dict:
    """调用 Gemini API 提取非结构化文本规则"""
    default_res = {
        "Cargo_forbidden": "unknown",
        "Volume_Limit": "unknown",
        "Pick_Packing_parcel": "unknown",
        "Tax_Policy": "unknown"
    }
    if not api_key or "YOUR_GEMINI" in api_key:
        return default_res

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        请阅读以下物流价格表中的说明文本，针对【目的地国家：{country}】提取并转译以下 4 个字段。
        必须严格输出 JSON 格式，不要包含 Markdown 符号：
        {{
            "Cargo_forbidden": "针对该国家的禁运物品列表，用英文逗号分隔",
            "Volume_Limit": "尺寸限制与公式，如 55x40x25",
            "Pick_Packing_parcel": "分拣打包仓储杂费数字，无提及填 unknown",
            "Tax_Policy": "清关模式(DDP/DDU)及 FOB/CIF 限制公式"
        }}
        说明文本：
        {text_context}
        """
        response = client.models.generate_content(
            model=model_name,  # 读取配置区 A 定的 Gemini 模型名称
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        st.warning(f"⚠️ AI 模型【{model_name}】解析【{country}】文本发生异常: {e}")
        return default_res

# =============================================================================
# 2. 核心算法模块 (0.25kg 阶梯重与边界修饰)
# =============================================================================
def generate_weight_steps(excel_w_min: float, excel_w_max: float) -> list:
    """生成 12 个固定 0.25kg 阶梯重并执行边界修饰"""
    standard_intervals = [
        (0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00),
        (1.00, 1.25), (1.25, 1.50), (1.50, 1.75), (1.75, 2.00),
        (2.00, 2.25), (2.25, 2.50), (2.50, 2.75), (2.75, 3.00)
    ]
    valid_steps = []
    for step_min, step_max in standard_intervals:
        if step_max <= excel_w_min or step_min >= excel_w_max:
            continue
        
        w_min = max(step_min, excel_w_min)
        w_max = min(step_max, excel_w_max)
        
        if excel_w_min >= 1 and w_min == step_min:
            w_min = 1.00
        elif excel_w_min > 1 and w_min == excel_w_min:
            w_min = round(excel_w_min + 0.01, 2)
            
        if excel_w_max <= 1 and w_max == step_max:
            w_max = 1.00
        elif excel_w_max < 1 and w_max == excel_w_max:
            w_max = round(excel_w_max - 0.01, 2)
            
        valid_steps.append((round(w_min, 2), round(w_max, 2)))
    return valid_steps

def parse_excel_weight_string(weight_str: str) -> tuple:
    """从 Excel 单元格解析起止重量"""
    if not isinstance(weight_str, str):
        return 0.0, 999.0
    s = weight_str.replace(' ', '').replace('kg', '').replace('KG', '')
    m = re.search(r'([\d\.]+)\s*<\s*W\s*[≤<=]\s*([\d\.]+)', s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.search(r'W\s*[≤<=]\s*([\d\.]+)', s)
    if m2:
        return 0.0, float(m2.group(1))
    return 0.0, 999.0

# =============================================================================
# 3. Excel 提取解析与数据拼接模块
# =============================================================================
def detect_supplier(filename: str) -> str:
    """匹配【配置区 B】自动识别人名/供应商"""
    fname_lower = filename.lower()
    for supplier_code, config in SUPPLIER_CONFIGS.items():
        if any(kw in fname_lower for kw in config["filename_keywords"]):
            return supplier_code
    return "Unknown"

def parse_supplier_excel(uploaded_file, supplier_code: str, api_key: str, model_name: str) -> pd.DataFrame:
    """读取 Excel 提取 14 个标准字段"""
    excel_file = pd.ExcelFile(uploaded_file)
    all_parsed_rows = []
    supp_cfg = SUPPLIER_CONFIGS.get(supplier_code, {})

    for sheet_name in excel_file.sheet_names:
        if any(skip_kw in sheet_name for skip_kw in ["目录", "对应表", "邮编", "禁运", "异形", "VAT"]):
            continue

        id_match = re.search(r'\(([A-Z0-9]+)\)', sheet_name)
        channel_id = id_match.group(1) if id_match else sheet_name.strip()
        cargo_category = "Regular" if "普货" in sheet_name else "Sensitive"

        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
        if df_raw.empty:
            continue

        header_idx = -1
        for idx, row in df_raw.iloc[:15].iterrows():
            row_str = "".join([str(v) for v in row.values if pd.notna(v)])
            if "国家" in row_str or "目的国" in row_str:
                header_idx = idx
                break
        if header_idx == -1:
            continue

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
        reg_col = next((c for c in headers if "挂号" in c or "处理" in c), None)
        time_col = next((c for c in headers if "时效" in c), None)

        if not country_col or not weight_col:
            continue

        ai_cache = {}

        for _, row in df_data.iterrows():
            country = str(row[country_col]).strip()
            if not country or country == "nan" or "说明" in country:
                continue

            raw_time = str(row[time_col]).strip() if time_col and pd.notna(row[time_col]) else "10-15 工作日"
            time_match = re.search(r'(\d+[-~]\d+|\d+)\s*(工作日|自然日|天)', raw_time)
            time_formatted = f"{time_match.group(1)} {time_match.group(2)}" if time_match else raw_time

            # 调用 AI（传入 key 与 model_name）
            if country not in ai_cache:
                ai_cache[country] = parse_unstructured_text_with_gemini(api_key, model_name, text_context, country)
            ai_info = ai_cache[country]

            src_w_min, src_w_max = parse_excel_weight_string(str(row[weight_col]))
            weight_steps = generate_weight_steps(src_w_min, src_w_max)

            r_kg = float(row[freight_col]) if freight_col and pd.notna(row[freight_col]) and str(row[freight_col]).replace('.','').isdigit() else 0.0
            r_parcel = float(row[reg_col]) if reg_col and pd.notna(row[reg_col]) and str(row[reg_col]).replace('.','').isdigit() else 0.0

            try:
                pick_pack = float(ai_info["Pick_Packing_parcel"])
            except:
                pick_pack = 0.0

            for w_min, w_max in weight_steps:
                total_rmb = round(w_max * r_kg + r_parcel + pick_pack, 2)

                all_parsed_rows.append({
                    "ID": channel_id,
                    "Destination Country": country,
                    "Cargo Category": cargo_category,
                    "Cargo forbidden": ai_info["Cargo_forbidden"],
                    "Time (workday/nature day)": time_formatted,
                    "Volume Limit (cm)": ai_info["Volume_Limit"],
                    "Volume to Weight parameter": volume_to_weight,
                    "Weight Range (min kg)": w_min,
                    "Weight Range (max kg)": w_max,
                    "RMB /kg": r_kg,
                    "RMB /parcel": r_parcel,
                    "Pick&Packing/parcel": ai_info["Pick_Packing_parcel"],
                    "RMB in total": total_rmb,
                    "Tax Policy": ai_info["Tax_Policy"]
                })

    return pd.DataFrame(all_parsed_rows)

# =============================================================================
# 4. Google Sheets 覆盖写回模块
# =============================================================================
def upsert_to_google_sheet(new_df: pd.DataFrame, spreadsheet_id: str, tab_name: str, primary_keys: list) -> int:
    """通过复合主键覆盖更新旧数据，追加新数据"""
    creds = json.loads(st.secrets["gcp_json"], strict=False)
    gc = gspread.service_account_from_dict(creds)
    sh = gc.open_by_key(spreadsheet_id)

    try:
        ws = sh.worksheet(tab_name)
        existing_data = ws.get_all_records()
        existing_df = pd.DataFrame(existing_data) if existing_data else pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows="2000", cols="20")
        existing_df = pd.DataFrame()

    if existing_df.empty:
        final_df = new_df
    else:
        existing_df['_pk'] = existing_df[primary_keys].astype(str).agg('_'.join, axis=1)
        new_df['_pk'] = new_df[primary_keys].astype(str).agg('_'.join, axis=1)

        filtered_existing = existing_df[~existing_df['_pk'].isin(new_df['_pk'])].copy()
        final_df = pd.concat([filtered_existing, new_df], ignore_index=True)
        final_df.drop(columns=['_pk'], errors='ignore', inplace=True)

    final_df_clean = final_df.fillna("")
    ws.clear()
    ws.update([final_df_clean.columns.values.tolist()] + final_df_clean.values.tolist())
    return len(final_df)


# =============================================================================
# 5. UI 界面布局
# =============================================================================
st.set_page_config(page_title="📦 多供应商物流报价 AI 解析系统", layout="wide")
st.title("📦 多供应商物流报价单 AI 解析系统")

with st.sidebar:
    st.header("控制面板")
    uploaded_file = st.file_uploader("上传报价单 (Excel)", type=["xlsx", "xls"])
    
    supplier_code = "Unknown"
    if uploaded_file:
        supplier_code = detect_supplier(uploaded_file.name)
        st.markdown(f"**识别供应商:** `{supplier_code}`")
        st.markdown(f"**调用模型:** `{GEMINI_MODEL}`")
    
    btn_start = st.button("🚀 开始解析并同步至云端", type="primary", disabled=(not uploaded_file or supplier_code == "Unknown"))

if not uploaded_file:
    st.info("👈 请先在左侧控制面板上传 Excel 报价单。")
else:
    if btn_start:
        with st.spinner(f"正在使用【{GEMINI_MODEL}】模型解析【{supplier_code}】报价单..."):
            # 传入配置区 A 的 GEMINI_API_KEY 和 GEMINI_MODEL
            parsed_df = parse_supplier_excel(uploaded_file, supplier_code, GEMINI_API_KEY, GEMINI_MODEL)
            
            if parsed_df.empty:
                st.error("❌ 未能解析出有效数据，请检查文件结构！")
            else:
                st.success(f"✅ 成功解析 {len(parsed_df)} 条阶梯价格记录！")
                st.dataframe(parsed_df.head(20), use_container_width=True)

                with st.spinner("正在比对复合主键并同步写入 Google Sheet..."):
                    total_count = upsert_to_google_sheet(parsed_df, SPREADSHEET_ID, MASTER_DB_TAB_NAME, COMPOSITE_PRIMARY_KEYS)
                    st.balloons()
                    st.success(f"🎉 同步成功！Google Sheet 云端【{MASTER_DB_TAB_NAME}】现存 {total_count} 条最新有效数据。")
