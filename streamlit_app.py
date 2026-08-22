import streamlit as st
import pandas as pd
import gspread
from openpyxl import load_workbook

st.set_page_config(page_title="物流报价解析系统", layout="wide")

# -----------------------------------------------------------------------------
# 0. 配置 Google Sheet ID (替换为你的网页 URL 里面 /d/ 和 /edit 之间的字符)
# -----------------------------------------------------------------------------
SPREADSHEET_ID = "你的_GOOGLE_SHEET_ID"

# -----------------------------------------------------------------------------
# 1. 核心控制逻辑：供应商识别与 Google Sheet 云端规则调取
# -----------------------------------------------------------------------------
def detect_supplier(uploaded_file) -> str:
    """仅根据上传文件名判断属于哪个供应商"""
    try:
        filename = uploaded_file.name.lower()
        
        signatures = {
            "4PX": ["递四方", "4px"],
            "YunExpress": ["云途", "yunexpress"],
            "SF": ["顺丰", "sf"]
        }
        
        for supplier, keywords in signatures.items():
            if any(kw in filename for kw in keywords):
                return supplier
                
        return "Unknown"
    except Exception as e:
        st.error(f"文件名读取失败: {e}")
        return "Unknown"

@st.cache_data(ttl=300)
def fetch_rules_from_gsheet(supplier: str):
    """从 Streamlit Cloud Secrets 加载凭证，调用 API 读取对应的 Tab 页规则"""
    if supplier == "Unknown":
        return None, False
        
    try:
        # ==================== Google Sheet API：从这里开始 ====================
        # 1. 从 Secrets 提取凭证字典
        creds = dict(st.secrets["gcp_service_account"])

        # 2. 自动把变形的 \n 修复为真正的换行
        if "private_key" in creds:
            creds["private_key"] = creds["private_key"].replace("\\n", "\n")

        # 3. 使用修复好的凭证初始化 API 客户端
        gc = gspread.service_account_from_dict(creds)
        # ==================== 放置位置：到这里结束 ====================
        
        # 4. 打开 Google Sheet 文档并定位到对应的 Tab 页
        sh = gc.open_by_key("1GjrPj2bKQZFz_ls5Y6ViI2fL_ovWcayN6ri58tiJErU")
        worksheet = sh.worksheet(supplier)
        
        # 5. 获取所有数据转为 DataFrame
        records = worksheet.get_all_records()
        rules_df = pd.DataFrame(records)
        return rules_df, True
        
    except gspread.exceptions.WorksheetNotFound:
        st.sidebar.error(f"❌ Google Sheet 中未找到名为 [{supplier}] 的 Tab 页！")
        return None, False
    except Exception as e:
        st.sidebar.error(f"❌ 读取云端规则 API 异常: {e}")
        return None, False

# -----------------------------------------------------------------------------
# 2. UI 界面布局
# -----------------------------------------------------------------------------
st.title("📦 物流报价单自动路由与解析系统")

# 侧边栏：文件上传与识别状态
with st.sidebar:
    st.header("文件控制台")
    uploaded_file = st.file_uploader("请上传报价单 (Excel)", type=["xlsx", "xls"])
    
    supplier = "Unknown"
    rules_df = None
    has_rule = False

    if uploaded_file:
        supplier = detect_supplier(uploaded_file)
        # 调用 API 拿到的不是文件路径，而是直接包含规则数据的 DataFrame
        rules_df, has_rule = fetch_rules_from_gsheet(supplier)
        
        st.divider()
        st.markdown(f"**识别供应商:** `{supplier}`")
        st.markdown(f"**云端规则 Tab:** `{supplier if has_rule else '未拉取到'}`")
        
        if has_rule:
            st.success("✅ 成功调取 Google Sheet 云端规则！")
        else:
            st.warning("⚠️ 暂未获得该供应商的解析规则。")

    parse_btn = st.button("开始运行解析", type="primary", disabled=(not has_rule))

# 主界面：内容展示
if not uploaded_file:
    st.info("👈 请在左侧侧边栏上传 Excel 报价单。")
else:
    tab1, tab2, tab3 = st.tabs(["📋 原始 Sheet 预览", "⚙️ 云端映射规则", "🚀 解析结果预测"])

    with tab1:
        st.subheader("Excel 包含的 Sheet 清单")
        excel_data = pd.ExcelFile(uploaded_file)
        st.write("所有 Sheet 名称：", excel_data.sheet_names)
        
        selected_sheet = st.selectbox("选择预览 Sheet", excel_data.sheet_names)
        if selected_sheet:
            df_preview = pd.read_excel(uploaded_file, sheet_name=selected_sheet, nrows=10)
            st.dataframe(df_preview, use_container_width=True)

    with tab2:
        st.subheader(f"[{supplier}] Google Sheet 实时映射规则")
        if has_rule and rules_df is not None:
            st.dataframe(rules_df, use_container_width=True)
        else:
            st.write("暂无规则数据。")

    with tab3:
        if parse_btn:
            st.subheader(f"使用云端 [{supplier}] 映射规则解析中...")
            # TODO: 此处后续直接把文件和 rules_df 传给具体的 parser：
            # parsed_df = run_parser(uploaded_file, rules_df)
            # st.dataframe(parsed_df)
            st.info("路由运行成功！已就绪 API 返回的规则数据，请在此处接入解析代码。")
        else:
            st.caption("请点击侧边栏的【开始运行解析】执行路由。")
