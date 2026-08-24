import io, json, re
import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI

# ========================= 配置 =========================
RULE_SHEET_ID = st.secrets["RULE_SHEET_ID"]
DATA_SHEET_ID = st.secrets["DATA_SHEET_ID"]
AI_API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.7-plus"
PRIMARY_KEYS = ["ID", "Destination Country", "Weight Range (max kg)"]
STANDARD_WEIGHTS = [(0,.25),(.25,.5),(.5,.75),(.75,1),(1,1.25),(1.25,1.5),(1.5,1.75),(1.75,2),(2,2.25),(2.25,2.5),(2.5,2.75),(2.75,3)]

# ========================= Google Sheets =========================
@st.cache_resource
def get_gsheet_client():
    creds = json.loads(st.secrets["gcp_json"], strict=False)
    scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    return gspread.authorize(ServiceAccountCredentials.from_json_keyfile_dict(creds, scopes))

def load_sheet(spreadsheet_id, worksheet_name):
    sh = get_gsheet_client().open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet_name)
    return pd.DataFrame(ws.get_all_records()), ws

def load_supplier_config():
    df, _ = load_sheet(RULE_SHEET_ID, "Supplier_Config")
    required = ["Supplier Code","Supplier Name","Enabled","Detection Type","Detection Value","Mapping Sheet"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Supplier_Config 缺少列：{', '.join(missing)}")
    return df

def load_mapping(sheet_name):
    df, _ = load_sheet(RULE_SHEET_ID, sheet_name)
    required = ["字段","是否AI读取","提取粒度","记录唯一键","Sheet定位类型","Sheet定位值","行定位类型","行定位值","列定位类型","列定位值","原始提取类型","Python解析器","AI指令","是否必填"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Mapping【{sheet_name}】缺少列：{', '.join(missing)}")
    return df

# ========================= 通用工具 =========================
def norm(v):
    if v is None or pd.isna(v): return ""
    return re.sub(r"\s+", " ", str(v).replace("\u3000", " ")).strip()

def enabled(v): return norm(v).lower() in {"true","1","yes","y","是"}

def rule(rules, field):
    x = rules[rules["字段"].astype(str).str.strip() == field]
    if x.empty: raise ValueError(f"Mapping 缺少字段：{field}")
    return x.iloc[0].to_dict()

def safe_float(v):
    if v is None or pd.isna(v): return None
    m = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(v))
    return float(m.group().replace(",", "")) if m else None

def match_rule(text, typ, val):
    text, val = norm(text), norm(val)
    if typ in ("none", ""): return True
    if typ == "exact": return text == val
    if typ == "contains": return val in text
    if typ == "regex": return bool(re.search(val, text, re.I))
    raise ValueError(f"不支持的定位类型：{typ}")

# ========================= 供应商识别 =========================
def detect_supplier(all_sheets):
    config = load_supplier_config()
    active = config[config["Enabled"].map(enabled)]
    matches = []
    for _, r in active.iterrows():
        score = sum(match_rule(s, r["Detection Type"], r["Detection Value"]) for s in all_sheets)
        if score: matches.append((score, r))
    if not matches: raise ValueError("无法识别供应商，请检查 Supplier_Config 的 Detection Rule。")
    matches.sort(key=lambda x: x[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ValueError(f"供应商识别冲突：{matches[0][1]['Supplier Code']} / {matches[1][1]['Supplier Code']}")
    r = matches[0][1]
    return norm(r["Supplier Code"]), norm(r["Supplier Name"]), norm(r["Mapping Sheet"])

# ========================= Excel =========================
@st.cache_data(show_spinner=False)
def load_excel(file_bytes):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)

def locate_sheets(all_sheets, r):
    return [s for s in all_sheets if match_rule(s, r["Sheet定位类型"], r["Sheet定位值"])]

def find_cell(df, typ, val, start_row=0):
    for rr in range(start_row, len(df)):
        for cc in range(df.shape[1]):
            t = norm(df.iat[rr, cc])
            if typ == "exact_header" and t == norm(val): return rr, cc
            if typ == "contains_header" and norm(val) in t: return rr, cc
            if typ == "exact_text" and t == norm(val): return rr, cc
            if typ == "contains_text" and norm(val) in t: return rr, cc
    return None

def find_column(df, r):
    x = find_cell(df, norm(r["列定位类型"]), norm(r["列定位值"]))
    if not x: raise ValueError(f"找不到列：{r['列定位类型']} / {r['列定位值']}")
    return x

def section_start(df):
    anchors = ["价格使用说明","计重规则","申报及税费"]
    for rr in range(len(df)):
        text = " ".join(norm(v) for v in df.iloc[rr].tolist() if norm(v))
        if any(a in text for a in anchors): return rr
    return len(df)

def find_country_rows(df, country_rule, target_country):
    header_row, country_col = find_column(df, country_rule)
    end_row = section_start(df)
    rows, current = [], ""
    for rr in range(header_row + 1, end_row):
        value = norm(df.iat[rr, country_col])
        if value: current = value
        if current == norm(target_country): rows.append(rr)
    return header_row, country_col, rows

def extract_section(df, anchor):
    start = None
    for rr in range(len(df)):
        text = " ".join(norm(v) for v in df.iloc[rr].tolist() if norm(v))
        if anchor in text:
            start = rr; break
    if start is None: return ""
    anchors = ["价格使用说明","计重规则","申报及税费"]
    out = []
    for rr in range(start, len(df)):
        text = " | ".join(norm(v) for v in df.iloc[rr].tolist() if norm(v))
        if not text: continue
        if rr > start and any(a in text for a in anchors if a != anchor): break
        out.append(f"Excel Row {rr + 1}: {text}")
    return "\n".join(out)

# ========================= 固定 Python 规则 =========================
def extract_id(sheet_name):
    m = re.search(r"[\(（]([A-Za-z0-9]+)[\)）]", sheet_name)
    if not m: raise ValueError(f"无法从 Sheet 名称提取 ID：{sheet_name}")
    return m.group(1)

def cargo_category(sheet_name):
    if "普货" in sheet_name: return "Regular"
    if any(x in sheet_name for x in ["带电","特货","敏感"]): return "Sensitive"
    return None

def generate_weights(source_min, source_max):
    if source_min is None or source_max is None or source_min >= source_max: return []
    out = []
    for smin, smax in STANDARD_WEIGHTS:
        if smax <= source_min or smin >= source_max: continue
        wmin, wmax = max(smin, source_min), min(smax, source_max)
        if source_min >= 1 and wmin == smin: wmin = 1.0
        elif source_min > 1 and wmin == source_min: wmin = round(source_min + .01, 2)
        if source_max <= 1 and wmax == smax: wmax = 1.0
        elif source_max < 1 and wmax == source_max: wmax = round(source_max - .01, 2)
        out.append((round(wmin,2), round(wmax,2)))
    return out

# ========================= AI =========================
@st.cache_data(show_spinner=False, max_entries=500)
def ai_json(prompt, context):
    client = OpenAI(api_key=AI_API_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(model=MODEL_NAME, temperature=0, messages=[
        {"role":"system","content":"你是严谨的物流报价表结构化提取专家。只能使用输入内容，不得猜测；无法确定返回null；必须返回合法JSON，不要输出Markdown。"},
        {"role":"user","content":f"{prompt}\n\n原始数据：\n{context}"}
    ])
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.I)
    return json.loads(raw)

def ai_metadata(target_country, country_context, note_text, weight_text, tax_text, time_cells, volume_cells, rules):
    fields = ["Cargo forbidden","Time (workday/nature day)","Volume Limit (cm)","Volume to Weight parameter","Pick&Packing/parcel","Tax Policy"]
    instructions = [f"{f}: {norm(rule(rules,f)['AI指令'])}" for f in fields if enabled(rule(rules,f)["是否AI读取"])]
    prompt = f'''目标国家：{target_country}\n\n根据以下AI指令提取：\n{chr(10).join(instructions)}\n\n严格返回JSON：{{"Cargo forbidden":[],"Time":{{"min":null,"max":null,"unit":null}},"Volume Limit":{{"length_cm":null,"width_cm":null,"height_cm":null,"max_length_cm":null,"max_volume_m3":null,"formula":null,"raw":null}},"Volume to Weight parameter":null,"Pick&Packing/parcel":null,"Tax Policy":{{"delivery_term":null,"fob_limit_usd":null,"cif_limit_usd":null,"raw":null}}}}\n\n只提取{target_country}；没有明确值返回null；不得猜测；Time只能使用提供的时效Cell；Volume Limit只能使用提供的标准尺寸Cell。'''
    context = f"目标国家价格行：\n{country_context}\n\n目标国家时效Cell：\n{time_cells}\n\n目标国家标准尺寸Cell：\n{volume_cells}\n\n价格使用说明：\n{note_text}\n\n计重规则：\n{weight_text}\n\n申报及税费：\n{tax_text}"
    return ai_json(prompt, context)

def ai_weight_map(target_country, weight_rows, weight_rule):
    prompt = f'''目标国家：{target_country}\n\n分析以下源重量区间。\n{norm(weight_rule["AI指令"])}\n\n标准目标max固定为：0.25、0.50、0.75、1.00、1.25、1.50、1.75、2.00、2.25、2.50、2.75、3.00。\n每个target_max_kg必须选择真实存在的source_excel_row；如果目标重量超过源最大计费重量，则返回null；不得创造行号。\n\n严格返回：{{"source_min_kg":null,"source_max_kg":null,"mapping":[{{"target_max_kg":0.25,"source_excel_row":12}}]}}'''
    return ai_json(prompt, json.dumps(weight_rows, ensure_ascii=False))

# ========================= AI结果格式化 =========================
def format_time(v):
    if not v: return None
    if isinstance(v, dict):
        mn, mx, unit = v.get("min"), v.get("max"), v.get("unit")
        if mn is None: return None
        return f"{mn} {unit}" if mx is None or mn == mx else f"{mn}~{mx} {unit}"
    return norm(v)

def format_dimension(v):
    if not v: return None
    if not isinstance(v, dict): return norm(v)
    p = []
    if all(v.get(k) is not None for k in ["length_cm","width_cm","height_cm"]): p.append(f"{v['length_cm']}×{v['width_cm']}×{v['height_cm']} cm")
    if v.get("max_length_cm") is not None: p.append(f"max_length={v['max_length_cm']}cm")
    if v.get("max_volume_m3") is not None: p.append(f"max_volume={v['max_volume_m3']}m³")
    if v.get("formula"): p.append(f"formula={v['formula']}")
    return "; ".join(p) or v.get("raw")

def format_tax(v):
    if not v: return None
    if not isinstance(v, dict): return norm(v)
    p = [v["delivery_term"]] if v.get("delivery_term") else []
    if v.get("fob_limit_usd") is not None: p.append(f"FOB < {v['fob_limit_usd']} USD")
    if v.get("cif_limit_usd") is not None: p.append(f"CIF < {v['cif_limit_usd']} USD")
    return ", ".join(p) or v.get("raw")

def format_forbidden(v):
    return ", ".join(norm(x) for x in v if norm(x)) if isinstance(v,list) else norm(v)

def pick_pack_value(v, weight_max):
    if isinstance(v, dict):
        by_weight = v.get("by_weight_max_kg", {})
        value = by_weight.get(str(round(weight_max,2)))
        return safe_float(value if value is not None else v.get("default"))
    return safe_float(v)

# ========================= 单线路解析 =========================
def parse_one_sheet(df, sheet_name, target_country, rules):
    rows, errors = [], []
    country_rule = rule(rules,"Destination Country")
    weight_rule = rule(rules,"Weight Range (max kg)")
    freight_rule = rule(rules,"RMB /kg")
    parcel_rule = rule(rules,"RMB /parcel")
    channel_id, cargo = extract_id(sheet_name), cargo_category(sheet_name)
    try:
        header_row, country_col, country_rows = find_country_rows(df,country_rule,target_country)
    except Exception as e:
        return rows,[{"Sheet":sheet_name,"Field":"Destination Country","Error":str(e)}]
    if not country_rows: return rows, errors

    time_rule = rule(rules,"Time (workday/nature day)")
    volume_rule = rule(rules,"Volume Limit (cm)")
    try:
        time_pos = find_column(df,time_rule)
    except Exception:
        time_pos = None
    try:
        volume_pos = find_column(df,volume_rule)
    except Exception:
        volume_pos = None

    time_cells, volume_cells = [], []
    if time_pos:
        for rr in country_rows:
            value = norm(df.iat[rr,time_pos[1]])
            if value: time_cells.append({"Excel Row":rr+1,"value":value})
    if volume_pos:
        for rr in country_rows:
            value = norm(df.iat[rr,volume_pos[1]])
            if value: volume_cells.append({"Excel Row":rr+1,"value":value})

    weight_pos = find_column(df,weight_rule)
    freight_pos = find_column(df,freight_rule)
    parcel_pos = find_column(df,parcel_rule)
    weight_col, freight_col, parcel_col = weight_pos[1], freight_pos[1], parcel_pos[1]

    country_context, weight_source_rows = [], []
    for rr in country_rows:
        vals = {f"Column_{c+1}":norm(df.iat[rr,c]) for c in range(df.shape[1]) if norm(df.iat[rr,c])}
        country_context.append({"Excel Row":rr+1,"Values":vals})
        weight_raw = norm(df.iat[rr,weight_col])
        if weight_raw: weight_source_rows.append({"source_excel_row":rr+1,"weight_range_raw":weight_raw,"freight_raw":norm(df.iat[rr,freight_col]),"parcel_raw":norm(df.iat[rr,parcel_col])})

    try:
        meta = ai_metadata(target_country,json.dumps(country_context,ensure_ascii=False),extract_section(df,"价格使用说明"),extract_section(df,"计重规则"),extract_section(df,"申报及税费"),json.dumps(time_cells,ensure_ascii=False),json.dumps(volume_cells,ensure_ascii=False),rules)
        wm = ai_weight_map(target_country,weight_source_rows,weight_rule)
    except Exception as e:
        return rows,[{"Sheet":sheet_name,"Field":"AI","Error":str(e)}]

    source_min, source_max = safe_float(wm.get("source_min_kg")), safe_float(wm.get("source_max_kg"))
    if source_min is None or source_max is None: return rows,[{"Sheet":sheet_name,"Field":"Weight Range","Error":"AI无法确定源重量范围"}]
    mapping = {}
    for x in wm.get("mapping",[]):
        mx, sr = safe_float(x.get("target_max_kg")), x.get("source_excel_row")
        if mx is not None and sr is not None: mapping[round(mx,2)] = int(sr)
    steps = generate_weights(source_min,source_max)
    pick_pack_raw = meta.get("Pick&Packing/parcel")
    valid_source_rows = {r+1 for r in country_rows}

    for wmin,wmax in steps:
        source_row = mapping.get(round(wmax,2))
        if source_row is None:
            errors.append({"Sheet":sheet_name,"Field":"Weight Range","Weight max":wmax,"Error":"AI没有指定对应源价格行"}); continue
        if source_row not in valid_source_rows:
            errors.append({"Sheet":sheet_name,"Field":"Weight Range","Weight max":wmax,"Error":f"AI指定的Excel Row {source_row}不属于目标国家"}); continue
        rr = source_row-1
        rkg, rparcel = safe_float(df.iat[rr,freight_col]), safe_float(df.iat[rr,parcel_col])
        ppack = pick_pack_value(pick_pack_raw,wmax)
        total = round(wmax*rkg+rparcel+ppack,2) if None not in [rkg,rparcel,ppack] else None
        rows.append({
            "ID":channel_id,"Destination Country":target_country,"Cargo Category":cargo,
            "Cargo forbidden":format_forbidden(meta.get("Cargo forbidden")),
            "Time (workday/nature day)":format_time(meta.get("Time")),
            "Volume Limit (cm)":format_dimension(meta.get("Volume Limit")),
            "Volume to Weight parameter":safe_float(meta.get("Volume to Weight parameter")),
            "Weight Range (min kg)":wmin,"Weight Range (max kg)":wmax,
            "RMB /kg":rkg,"RMB /parcel":rparcel,"Pick&Packing/parcel":ppack,
            "RMB in total":total,"Tax Policy":format_tax(meta.get("Tax Policy"))
        })
    return rows, errors

# ========================= 总解析 =========================
def parse_workbook(all_sheets,target_country,rules):
    sheets = locate_sheets(all_sheets,rule(rules,"ID"))
    if not sheets: raise ValueError("没有找到符合当前供应商 Mapping 的线路 Sheet。")
    all_rows, errors = [], []
    progress, status = st.progress(0), st.empty()
    for i,sheet_name in enumerate(sheets,1):
        status.markdown(f"**解析 [{i}/{len(sheets)}]** `{sheet_name}` → `{target_country}`")
        try:
            r,e = parse_one_sheet(all_sheets[sheet_name],sheet_name,target_country,rules); all_rows.extend(r); errors.extend(e)
        except Exception as ex:
            errors.append({"Sheet":sheet_name,"Field":"Parser","Error":str(ex)})
        progress.progress(i/len(sheets))
    progress.empty(); status.success("✅ 解析完成")
    result, errdf = pd.DataFrame(all_rows), pd.DataFrame(errors)
    if not result.empty: result = result.drop_duplicates(subset=PRIMARY_KEYS,keep="last").reset_index(drop=True)
    return result, errdf

# ========================= 历史数据 / 更新 =========================
def get_country_ws(country):
    sh = get_gsheet_client().open_by_key(DATA_SHEET_ID)
    try: return sh.worksheet(country)
    except gspread.exceptions.WorksheetNotFound: return sh.add_worksheet(title=country,rows="2000",cols="30")

def compare_data(new_df,old_df):
    if old_df.empty: return {"new":new_df.copy(),"updated":pd.DataFrame(),"unchanged":pd.DataFrame(),"final":new_df.copy()}
    new, old = new_df.copy(), old_df.copy()
    for c in PRIMARY_KEYS: new[c], old[c] = new.get(c,""), old.get(c,"")
    new["_pk"] = new[PRIMARY_KEYS].astype(str).agg("|".join,axis=1)
    old["_pk"] = old[PRIMARY_KEYS].astype(str).agg("|".join,axis=1)
    old = old.drop_duplicates("_pk",keep="last"); old_map = old.set_index("_pk",drop=False)
    nrows, urows, irows = [], [], []
    for _,n in new.iterrows():
        pk=n["_pk"]
        if pk not in old_map.index: nrows.append(n.drop("_pk").to_dict()); continue
        o=old_map.loc[pk]
        changed=any(norm(n.get(c,""))!=norm(o.get(c,"")) for c in new.columns if c!="_pk" and c in old.columns)
        (urows if changed else irows).append(n.drop("_pk").to_dict())
    untouched=old[~old["_pk"].isin(new["_pk"])].drop(columns="_pk",errors="ignore")
    final=pd.concat([untouched,pd.DataFrame(nrows),pd.DataFrame(urows),pd.DataFrame(irows)],ignore_index=True)
    return {"new":pd.DataFrame(nrows),"updated":pd.DataFrame(urows),"unchanged":pd.DataFrame(irows),"final":final}

def write_data(ws,df):
    df=df.fillna(""); ws.clear(); ws.update([df.columns.tolist()]+df.astype(str).values.tolist(),range_name="A1"); return len(df)

# ========================= App =========================
st.title("📦 物流报价规则解析器")
st.caption("自动识别供应商 → 加载对应 Mapping → 指定国家 → Python + AI 提取 → 预览 → 更新")
st.subheader("① 上传报价表")
uploaded_file=st.file_uploader("把供应商报价 Excel 拖到这里",type=["xlsx","xls"])
st.subheader("② 输入目标国家/地区")
target_country=st.text_input("目标国家/地区",placeholder="例如：墨西哥、美国、加拿大").strip()
if uploaded_file: st.info(f"已选择文件：{uploaded_file.name}")

run=st.button("🚀 识别供应商并开始解析",type="primary",use_container_width=True,disabled=not uploaded_file or not target_country)
if run:
    try:
        with st.spinner("正在读取 Excel 并识别供应商..."):
            all_sheets=load_excel(uploaded_file.getvalue())
            supplier_code,supplier_name,mapping_sheet=detect_supplier(all_sheets)
            rules=load_mapping(mapping_sheet)
        st.success(f"✅ 供应商：{supplier_name}（{supplier_code}）")
        st.info(f"✅ Mapping：{mapping_sheet}；目标国家：{target_country}")

        with st.spinner(f"正在解析【{target_country}】..."):
            parsed_df,errors_df=parse_workbook(all_sheets,target_country,rules)
        if parsed_df.empty:
            st.error(f"❌ 没有提取到【{target_country}】的数据。")
            if not errors_df.empty: st.dataframe(errors_df,use_container_width=True)
            st.stop()

        ws=get_country_ws(target_country); old_df=pd.DataFrame(ws.get_all_records()); comparison=compare_data(parsed_df,old_df)
        c1,c2,c3,c4=st.columns(4); c1.metric("解析记录",len(parsed_df)); c2.metric("新增",len(comparison["new"])); c3.metric("更新",len(comparison["updated"])); c4.metric("异常",len(errors_df))
        t1,t2,t3,t4=st.tabs(["全部结果","新增","更新","异常"])
        with t1: st.dataframe(parsed_df,use_container_width=True,height=600)
        with t2: st.dataframe(comparison["new"],use_container_width=True)
        with t3: st.dataframe(comparison["updated"],use_container_width=True)
        with t4:
            if errors_df.empty: st.success("✅ 没有发现异常")
            else: st.dataframe(errors_df,use_container_width=True)
        st.download_button("⬇️ 下载解析结果 CSV",parsed_df.to_csv(index=False).encode("utf-8-sig"),file_name=f"{supplier_code}_{target_country}.csv",mime="text/csv")

        st.warning(f"确认后将更新 Google Sheet【{target_country}】；唯一键：ID + Destination Country + Weight Range (max kg)。")
        confirm=st.checkbox("我确认解析结果，执行更新")
        if st.button("✅ 确认并更新 Google Sheet",type="primary",use_container_width=True,disabled=not confirm):
            count=write_data(ws,comparison["final"]); st.success(f"🎉 更新完成，共 {count} 条记录。")
    except Exception as e:
        st.error("❌ 运行失败")
        st.exception(e)
