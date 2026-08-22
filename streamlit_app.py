import streamlit as st
from openpyxl import load_workbook


# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="Logistics System",
    page_icon="📦",
    layout="wide"
)


# =========================
# 页面标题
# =========================
st.title("📦 Logistics Rate Management System")
st.caption("运输报价解析与管理系统")


# =========================
# 文件上传
# =========================
st.subheader("1. 上传运输报价表")

uploaded_file = st.file_uploader(
    "选择 Excel 文件",
    type=["xlsx", "xls"],
    help="上传物流供应商提供的运输报价表"
)


# =========================
# 供应商识别
# =========================
def detect_supplier(workbook):
    """
    根据 Excel 的 Sheet 名称和内容判断供应商。
    第一版只支持 4PX。
    """

    sheet_names = workbook.sheetnames

    # 4PX 的特征
    has_4px_sheet = any(
        "联邮通" in sheet_name
        for sheet_name in sheet_names
    )

    has_4px_vat = any(
        "VAT税率参照表" in sheet_name
        for sheet_name in sheet_names
    )

    has_4px_directory = any(
        "价格表目录" in sheet_name
        for sheet_name in sheet_names
    )

    if has_4px_sheet and (has_4px_vat or has_4px_directory):
        return "4PX"

    return None


# =========================
# 上传后处理
# =========================
if uploaded_file is not None:

    st.success(f"文件已上传：{uploaded_file.name}")

    try:
        # openpyxl 读取 Excel
        workbook = load_workbook(
            uploaded_file,
            read_only=True,
            data_only=True
        )

        # 显示 Sheet
        st.subheader("2. 文件信息")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Sheet 数量",
                len(workbook.sheetnames)
            )

        with col2:
            st.metric(
                "文件大小",
                f"{uploaded_file.size / 1024:.1f} KB"
            )

        # 供应商识别
        st.subheader("3. 供应商识别")

        supplier = detect_supplier(workbook)

        if supplier:
            st.success(f"识别结果：{supplier}")

            st.info(
                f"系统将使用 `{supplier}` 的 Mapping 规则进行解析。"
            )

        else:
            st.error(
                "无法识别供应商。暂时只支持 4PX。"
            )

        # Sheet 列表
        st.subheader("4. Excel Sheet")

        for i, sheet_name in enumerate(workbook.sheetnames, start=1):
            st.write(f"{i}. {sheet_name}")

    except Exception as e:
        st.error(f"Excel 读取失败：{e}")
