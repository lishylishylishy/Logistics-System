import os
import streamlit as st
import pandas as pd
from openpyxl import load_workbook

st.set_page_config(page_title="物流报价解析系统", layout="wide")

# -----------------------------------------------------------------------------
# 1. 核心控制逻辑：供应商识别与规则路由
# -----------------------------------------------------------------------------
def detect_supplier(uploaded_file) -> str:
    """第一步：根据 Excel Sheet 名称判断属于哪个供应商"""
    try:
        wb = load_workbook(uploaded_file, read_only=True, data_only=True)
        sheet_names = wb.sheetnames
        
        # 供应商特征映射表（仅做身份识别）
        signatures = {
            "4PX": ["递四方","4PX"],
            "YunExpress": ["云途", "YunExpress"],
            "SF": ["顺丰", "SFExpress"]
        }
        
        for supplier, keywords in signatures.items():
            for sheet in sheet_names:
                if any(kw in sheet for kw in keywords):
                    return supplier
        return "Unknown"
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return "Unknown"

def get_mapping_rule_path(supplier: str):
    """第二步：根据供应商名称，自动路由到对应的映射配置文件"""
    if supplier == "Unknown":
        return None, False
    
    # 约定好的规则文件存储路径，例如：mappings/4px/mapping.yaml
    rule_path = f"mappings/{supplier.lower()}/mapping.yaml"
    has_rule = os.path.exists(rule_path)
    return rule_path, has_rule

# -----------------------------------------------------------------------------
# 2. UI 界面布局
# -----------------------------------------------------------------------------
st.title("📦 物流报价单自动路由与解析系统")

# 侧边栏：文件上传与识别状态
with st.sidebar:
    st.header("文件控制台")
    uploaded_file = st.file_uploader("请上传报价单 (Excel)", type=["xlsx", "xls"])
    
    supplier = "Unknown"
    rule_path = None
    has_rule = False

    if uploaded_file:
        supplier = detect_supplier(uploaded_file)
        rule_path, has_rule = get_mapping_rule_path(supplier)
        
        st.divider()
        st.markdown(f"**识别供应商:** `{supplier}`")
        st.markdown(f"**路由配置文件:** `{rule_path}`")
        
        if has_rule:
            st.success("✅ 找到匹配的映射规则！")
        else:
            st.warning("⚠️ 暂未找到该供应商的映射规则文件。")

    parse_btn = st.button("开始运行解析", type="primary", disabled=(not has_rule))

# 主界面：内容展示
if not uploaded_file:
    st.info("👈 请在左侧侧边栏上传 Excel 报价单。")
else:
    tab1, tab2 = st.tabs(["📋 原始 Sheet 预览", "🚀 解析结果预测"])

    with tab1:
        st.subheader("Excel 包含的 Sheet 清单")
        excel_data = pd.ExcelFile(uploaded_file)
        st.write("所有 Sheet 名称：", excel_data.sheet_names)
        
        selected_sheet = st.selectbox("选择预览 Sheet", excel_data.sheet_names)
        if selected_sheet:
            df_preview = pd.read_excel(uploaded_file, sheet_name=selected_sheet, nrows=10)
            st.dataframe(df_preview, use_container_width=True)

    with tab2:
        if parse_btn:
            st.subheader(f"使用 [{rule_path}] 运行解析中...")
            # TODO: 此处后续只需一行代码调用你的通用解析器：
            # parsed_df = run_parser(uploaded_file, rule_path)
            # st.dataframe(parsed_df)
            st.info("路由运行成功！请在此处接入具体 parser 的返回数据。")
        else:
            st.caption("请点击侧边栏的【开始运行解析】执行路由。")
