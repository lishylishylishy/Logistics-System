import streamlit as st
import pandas as pd
import re
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI
import io

# ==========================================
# ⚙️ 第一部分：核心配置区域 (请修改这里！)
# ==========================================

# 1. 智谱 API 配置
ZHIPU_API_KEY = st.secrets.get("ZHIPU_API_KEY", "把你的智谱API密钥填在这里") 
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL_NAME = "glm-4-flash"  # 这个模型完全免费

# 2. Google Sheet 数据库配置
GOOGLE_SHEET_ID = "请在这里填入你刚刚新建的Google_Sheet_ID"
GOOGLE_SHEET_TAB_NAME = "Sheet1" # 默认新建的工作表名称，如果改了请同步修改

# ==========================================

# 设置页面宽度
st.set_page_config(layout="wide", page_title="物流价格表解析系统")

def safe_float(val):
    """安全地从单元格中提取费用数字"""
    if pd.isna(val): 
        return 0.0
    val_str = str(val).strip()
    # 提取数字和小数点
    nums = re.findall(r'[\d\.]+', val_str)
    if nums:
        try:
            return float(nums[0])
        except:
            pass
    return 0.0

def parse_with_zhipu(text_context, country):
    """调用智谱免费大模型提取非结构化规则"""
    default_res = {
        "Cargo_forbidden": "unknown",
        "Volume_Limit": "unknown",
        "Pick_Packing_parcel": "unknown",
        "Tax_Policy": "unknown"
    }
    
    if not ZHIPU_API_KEY or ZHIPU_API_KEY.startswith("把你"):
        return default_res

    try:
        client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)
        prompt = f"""
        请阅读以下物流说明文本，针对【目的地国家：{country}】提取以下 4 个字段。
        必须严格输出 JSON 格式，不要有 Markdown 符号：
        {{
            "Cargo_forbidden": "针对该国家的禁运物品，用逗号分隔",
            "Volume_Limit": "尺寸限制，如 55x40x25",
            "Pick_Packing_parcel": "分拣/操作费数字，无提及填 unknown",
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
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content
        # 简单清理可能包含的 markdown 标签
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        return json.loads(result_text)
    except Exception as e:
        st.warning(f"AI解析【{country}】报错: {e}")
        return default_res

def parse_supplier_excel(file_bytes, file_name):
    """核心解析逻辑"""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    all_data = []
    
    for sheet_name in xls.sheet_names:
        # 1. 修复ID提取逻辑（兼容中文括号）
        id_match = re.search(r'[\(（]([A-Za-z0-9]+)[\)）]', sheet_name)
        channel_id = id_match.group(1) if id_match else sheet_name
        
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        
        header_row_idx = None
        for idx, row in df.iterrows():
            row_str = " ".join(row.astype(str).tolist())
            if "国家" in row_str and ("运费" in row_str or "KG" in row_str.upper()):
                header_row_idx = idx
                break
        
        if header_row_idx is None:
            continue
            
        df_data = df.iloc[header_row_idx+1:].copy()
        headers = df.iloc[header_row_idx].astype(str).tolist()
        df_data.columns = headers
        
        # 提取整表的说明文本给 AI 用
        full_text = " ".join(df.astype(str).values.flatten())
        
        country_col = next((col for col in headers if "国家" in str(col)), None)
        freight_col = next((col for col in headers if "运费" in str(col) or "KG" in str(col).upper()), None)
        reg_col = next((col for col in headers if "挂号费" in str(col) or "处理费" in str(col) or "件" in str(col)), None)
        
        if not country_col:
            continue
            
        for _, row in df_data.iterrows():
            country = str(row[country_col]).strip()
            
            # 2. 修复国家字段错位（跳过空值和底部的备注说明行）
            if not country or country.lower() == "nan" or any(kw in country for kw in ["说明", "承诺", "注", "以上", "计费", ":", "：", "生效"]):
                continue
                
            # 3. 修复费用提取报错
            r_kg = safe_float(row[freight_col]) if freight_col else 0.0
            r_parcel = safe_float(row[reg_col]) if reg_col else 0.0
            
            # 调用 AI 提取非结构化字段
            ai_data = parse_with_zhipu(full_text[:3000], country)
            
            all_data.append({
                "ID": channel_id,
                "Destination Country": country,
                "RMB /kg": r_kg,
                "RMB /parcel": r_parcel,
                "Cargo forbidden": ai_data.get("Cargo_forbidden", "unknown"),
                "Volume Limit (cm)": ai_data.get("Volume_Limit", "unknown"),
                "Pick&Packing/parcel": ai_data.get("Pick_Packing_parcel", "unknown"),
                "Tax Policy": ai_data.get("Tax_Policy", "unknown")
            })
            
    return pd.DataFrame(all_data)

def write_to_gsheet(df):
    """写入 Google Sheet"""
    try:
        # 从 Secrets 读取 GCP 凭证
        gcp_cred_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            dict(gcp_cred_dict),
            ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_TAB_NAME)
        
        # 清空原表并写入新数据
        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())
        return True
    except Exception as e:
        st.error(f"写入 Google Sheet 失败: {e}")
        return False

# ==========================================
# 🖥️ 页面 UI 逻辑与状态管理
# ==========================================

st.title("📦 物流报价表智能解析系统")
st.markdown("使用 **智谱大模型 (GLM-4-Flash)** 免费解析非结构化规则，并同步至 Google Sheet。")

# 初始化 Session State (缓存状态)
if "parsed_df" not in st.session_state:
    st.session_state.parsed_df = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

uploaded_file = st.file_uploader("上传 4PX 报价表 (Excel格式)", type=["xlsx", "xls"])

if uploaded_file is not None:
    # 只有当上传了新文件时，才触发重新解析
    if uploaded_file.name != st.session_state.uploaded_file_name:
        with st.spinner("🚀 正在解析 Excel 数据并调用大模型提取规则，请稍候..."):
            file_bytes = uploaded_file.read()
            df = parse_supplier_excel(file_bytes, uploaded_file.name)
            
            # 保存到缓存
            st.session_state.parsed_df = df
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success("解析完成！数据已缓存。")

# 如果缓存中有数据，展示出来
if st.session_state.parsed_df is not None:
    st.subheader("📊 解析结果预览")
    df_to_show = st.session_state.parsed_df
    
    # 在主页直接展示数据表
    st.dataframe(df_to_show, use_container_width=True)
    
    if st.button("💾 将数据同步到 Google Sheet"):
        with st.spinner("正在同步..."):
            if write_to_gsheet(df_to_show):
                st.success("✅ 同步成功！前往你的 Google Sheet 即可查看。")
