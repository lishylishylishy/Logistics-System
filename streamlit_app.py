import streamlit as st
import pandas as pd
from openpyxl import load_workbook
import yaml

st.set_page_config(
    page_title="物流报价解析系统",
    page_icon="📦",
    layout="wide"
)

# 自定义微调 CSS：控制标题字号与间距
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.8rem !important; font-weight: 600; }
    h2 { font-size: 1.3rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- 侧边栏：文件上传与控制中心 ---
with st.sidebar:
    st.title("📦 物流报价解析系统")
    st.caption("自动化报价表解析与数据库同步")
    st.divider()
    
    uploaded_file = st.file_uploader(
        "上传报价单 Excel",
        type=["xlsx", "xls"],
        help="支持 4PX、云途等通用物流报价表"
    )
    
    supplier = None
    if uploaded_file is not None:
        try:
            wb = load_workbook(uploaded_file, read_only=True, data_only=True)
            sheet_names = wb.sheetnames
            
            # 识别供应商逻辑
            if any("联邮通" in name for name in sheet_names) or any("价格表目录" in name for name in sheet_names):
                supplier = "4PX"
            
            if supplier:
                st.success(f"识别供应商: **{supplier}**")
            else:
                st.error("无法自动识别供应商")
        except Exception as e:
            st.error(f"读取文件失败: {e}")

    st.divider()
    parse_btn = st.button("🚀 开始自动化解析", type="primary", use_container_width=True, disabled=(not supplier))

# --- 主界面：解析结果与查看 ---
st.title("报价数据看板")

if not uploaded_file:
    st.info("👈 请在左侧侧边栏拖入 Excel 报价表开始操作。")
else:
    # 顶部指标卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sheet 总数", len(sheet_names))
    col2.metric("识别结果", supplier if supplier else "未知")
    col3.metric("解析状态", "待解析" if not parse_btn else "处理完成")
    col4.metric("异常排查项", "0 项" if not parse_btn else "需人工确认 2 项")

    # 分页展示区
    tab1, tab2, tab3 = st.tabs(["📊 标准化价格表", "⚠️ 异常/政策提示 (AI)", "📋 原始 Sheet 预览"])
    
    with tab1:
        if parse_btn:
            st.subheader("标准化结构数据预览")
            # 占位演示数据，后续对接 parser.py
            sample_df = pd.DataFrame([
                {"carrier": supplier, "country": "英国", "product_code": "ZQ", "weight_range": "0-2kg", "freight_price": 100, "registration_fee": 21, "tax_included": "NO"},
                {"carrier": supplier, "country": "法国", "product_code": "ZQ", "weight_range": "0-2kg", "freight_price": 105, "registration_fee": 22, "tax_included": "NO"},
            ])
            st.dataframe(sample_df, use_container_width=True)
            st.button("⬆️ 确认同步至 Google Sheets")
        else:
            st.caption("点击左侧侧边栏【开始自动化解析】按钮查看解析提取结果。")
            
    with tab2:
        st.subheader("AI 提取的附加费与税费政策")
        st.write("- **超尺寸附加费**: 最长边 > 60cm 需加收 130 RMB/票")
        st.write("- **VAT 增值税**: 未含税，按目的地国家实际税率核算")

    with tab3:
        st.subheader("原始 Sheet 清单")
        st.dataframe(pd.DataFrame({"Sheet 名称": sheet_names}), height=300, use_container_width=True)
