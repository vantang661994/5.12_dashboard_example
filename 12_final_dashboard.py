# final_dashboard.py
# 📊 Final Project Dashboard – Streamlit + Plotly
# Tính năng:
# - Nhiều dataset: Stock (yfinance), Sales (giả lập), Student (giả lập), Upload CSV
# - ≥3 biểu đồ/bảng: line/bar/pie/box/heatmap + data table
# - Widget tương tác: selectbox/radio/slider/date_input/multiselect/text input
# - KPI Cards, Tabs, Download CSV, Cache dữ liệu, Clean layout

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime, timedelta

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="📊 Final Project Dashboard", layout="wide")
st.title("📊 Final Project: Build Your Own Dashboard")
st.caption("Module 6 • Bài 17–18 • Tích hợp tất cả kiến thức: widgets, charts, tables, thiết kế sạch.")

# ========== UTIL & CACHE ==========
@st.cache_data(show_spinner=True)
def load_stock_prices(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

@st.cache_data(show_spinner=False)
def synthesize_sales(n_days: int = 365, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(datetime.today() - timedelta(days=n_days-1), periods=n_days, freq="D")
    regions = ["North", "South", "East", "West"]
    products = ["A", "B", "C", "D"]

    rows = []
    for d in dates:
        for r in regions:
            for p in products:
                price = {"A": 25, "B": 45, "C": 60, "D": 80}[p]
                base = rng.normal(40, 12)
                season = 1.2 if d.month in (11, 12) else (0.9 if d.month in (6, 7) else 1.0)
                qty = max(0, int(base * season + rng.normal(0, 5)))
                revenue = qty * price
                rows.append((d, r, p, qty, price, revenue))
    df = pd.DataFrame(rows, columns=["Date", "Region", "Product", "Qty", "Price", "Revenue"])
    return df

@st.cache_data(show_spinner=False)
def synthesize_student(n_students: int = 400, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genders = rng.choice(["Male", "Female"], size=n_students)
    classes = rng.choice(["10A", "10B", "11A", "11B", "12A", "12B"], size=n_students)
    math = np.clip(rng.normal(75, 12, n_students), 0, 100)
    reading = np.clip(rng.normal(78, 11, n_students), 0, 100)
    writing = np.clip(rng.normal(76, 10, n_students), 0, 100)
    attend = np.clip(rng.normal(92, 6, n_students), 60, 100)

    df = pd.DataFrame({
        "StudentID": np.arange(1, n_students + 1),
        "Gender": genders,
        "Class": classes,
        "Math": math.round(1),
        "Reading": reading.round(1),
        "Writing": writing.round(1),
        "Attendance": attend.round(1),
    })
    df["Average"] = df[["Math", "Reading", "Writing"]].mean(axis=1).round(1)
    return df

def kpi_card(label: str, value, delta=None, help_text=None):
    col = st.container()
    with col:
        st.metric(label, value, delta=delta, help=help_text)

def download_button(df: pd.DataFrame, label="⬇️ Download CSV", filename="data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ========== SIDEBAR – DATASET PICKER ==========
with st.sidebar:
    st.header("⚙️ Settings")
    dataset = st.selectbox(
        "Dataset",
        ["Stock Prices (Yahoo Finance)", "Sales Data (Synthetic)", "Student Performance (Synthetic)", "Upload CSV"]
    )
    show_help = st.checkbox("Show inline help", value=True)

# ========== DATASET: STOCK ==========
if dataset == "Stock Prices (Yahoo Finance)":
    with st.sidebar:
        st.subheader("Stock Filters")
        symbol = st.text_input("Ticker (e.g., AAPL, MSFT, TSLA)", value="AAPL").strip().upper()
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
        roll = st.slider("Rolling Window (days)", 1, 60, 14)

    df = load_stock_prices(symbol, period=period, interval=interval)

    if isinstance(df.columns, pd.MultiIndex):
    # If columns look like ('Close','AAPL'), ('Volume','AAPL'), ...
        # keep only the field level for the selected symbol
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)   # now columns: Open, High, Low, Close, Volume, ...
        else:
            # Fallback: flatten by joining the tuple pieces
            df.columns = ['_'.join(map(str, c)).strip() for c in df.columns]

    if df.empty:
        st.warning("Không tải được dữ liệu. Hãy thử mã khác hoặc khoảng thời gian khác.")
    else:
        # KPIs
        latest = df["Close"].dropna().iloc[-1].item()
        prev   = df["Close"].dropna().iloc[-2].item()
        delta  = f"{(latest - prev) / prev * 100:.2f}%"

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Symbol", symbol)
        with c2: kpi_card("Last Close", f"${latest:,.2f}", delta=delta)
        with c3: kpi_card("Period bars", len(df))
        with c4: kpi_card("Interval", interval)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Price", "📊 Volume & Returns", "📄 Data Table"])
        with tab1:
            df["SMA"] = df["Close"].rolling(roll).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], mode="lines", name=f"SMA({roll})"))
            fig.update_layout(title=f"{symbol} – Close & SMA",
                              xaxis_title="Date", yaxis_title="Price (USD)", legend_title="Series")
            st.plotly_chart(fig, use_container_width=True)

            # High-Low band
            band = go.Figure()
            band.add_trace(go.Scatter(x=df.index, y=df["High"], name="High", mode="lines"))
            band.add_trace(go.Scatter(x=df.index, y=df["Low"], name="Low", mode="lines", fill='tonexty'))
            band.update_layout(title=f"{symbol} – High/Low Band", xaxis_title="Date", yaxis_title="USD")
            st.plotly_chart(band, use_container_width=True)

        with tab2:
            df_ret = df[["Close", "Volume"]].copy()
            df_ret["Return(%)"] = df_ret["Close"].pct_change() * 100
            cL, cR = st.columns(2)
            with cL:
                figv = px.bar(df_ret, x=df_ret.index, y="Volume", title="Trading Volume")
                st.plotly_chart(figv, use_container_width=True)
            with cR:
                figr = px.histogram(df_ret.dropna(), x="Return(%)", nbins=40, title="Return Distribution (%)")
                st.plotly_chart(figr, use_container_width=True)

        with tab3:
            st.dataframe(df.reset_index().tail(200))
            download_button(df.reset_index(), filename=f"{symbol}_{period}_{interval}.csv")

        if show_help:
            st.info("Gợi ý: Dùng `symbol`, `period`, `interval`, `rolling` để khám phá hành vi giá. "
                    "Bạn đã có >=3 biểu đồ (Close+SMA, High/Low Band, Volume, Histogram) + Data Table + KPI.")

# ========== DATASET: SALES ==========
elif dataset == "Sales Data (Synthetic)":
    with st.sidebar:
        st.subheader("Sales Filters")
        days = st.slider("Số ngày", 60, 540, 365)
        regions_sel = st.multiselect("Regions", ["North", "South", "East", "West"], default=["North", "South", "East", "West"])
        products_sel = st.multiselect("Products", ["A", "B", "C", "D"], default=["A", "B", "C", "D"])
        agg_level = st.radio("Aggregate by", ["Date", "Region", "Product"], horizontal=True)

    df = synthesize_sales(n_days=days)
    df = df[df["Region"].isin(regions_sel) & df["Product"].isin(products_sel)]

    # KPIs
    total_rev = df["Revenue"].sum()
    avg_price = df["Price"].mean()
    total_qty = df["Qty"].sum()
    unique_skus = df["Product"].nunique()
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Revenue", f"${total_rev:,.0f}")
    with c2: kpi_card("Total Quantity", f"{total_qty:,}")
    with c3: kpi_card("Avg Price", f"${avg_price:,.2f}")
    with c4: kpi_card("Products", unique_skus)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Trend", "🧭 Breakdown", "📄 Table & Export"])
    with tab1:
        if agg_level == "Date":
            dfa = df.groupby("Date", as_index=False).agg({"Revenue":"sum","Qty":"sum"})
            fig = px.line(dfa, x="Date", y="Revenue", title="Revenue Over Time")
        elif agg_level == "Region":
            dfa = df.groupby(["Date","Region"], as_index=False).agg({"Revenue":"sum"})
            fig = px.line(dfa, x="Date", y="Revenue", color="Region", title="Revenue by Region")
        else:
            dfa = df.groupby(["Date","Product"], as_index=False).agg({"Revenue":"sum"})
            fig = px.line(dfa, x="Date", y="Revenue", color="Product", title="Revenue by Product")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cL, cR = st.columns(2)
        with cL:
            by_region = df.groupby("Region", as_index=False).agg({"Revenue":"sum", "Qty":"sum"})
            fig1 = px.bar(by_region, x="Region", y="Revenue", title="Revenue by Region", text_auto=True)
            st.plotly_chart(fig1, use_container_width=True)
        with cR:
            by_product = df.groupby("Product", as_index=False).agg({"Revenue":"sum"})
            fig2 = px.pie(by_product, names="Product", values="Revenue", title="Revenue Share by Product", hole=0.3)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.dataframe(df.tail(1000))
        download_button(df, filename="sales_data.csv")

    if show_help:
        st.info("Gợi ý: Trang này có line chart (trend), bar/pie (breakdown) và table + download. "
                "Bạn có thể đổi aggregate level để luyện tư duy thiết kế.")

# ========== DATASET: STUDENT ==========
elif dataset == "Student Performance (Synthetic)":
    with st.sidebar:
        st.subheader("Student Filters")
        n = st.slider("Số học sinh", 100, 1200, 400, step=50)
        classes = ["10A","10B","11A","11B","12A","12B"]
        class_sel = st.multiselect("Lớp", classes, default=classes)
        gender_sel = st.multiselect("Giới tính", ["Male","Female"], default=["Male","Female"])
        score_field = st.selectbox("Môn học", ["Math", "Reading", "Writing", "Average"], index=3)

    df = synthesize_student(n_students=n)
    df = df[df["Class"].isin(class_sel) & df["Gender"].isin(gender_sel)]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Số HS", len(df))
    with c2: kpi_card(f"Điểm TB ({score_field})", f"{df[score_field].mean():.1f}")
    with c3: kpi_card("Chuyên cần TB", f"{df['Attendance'].mean():.1f}%")
    with c4: kpi_card("Số lớp", df["Class"].nunique())

    tab1, tab2, tab3 = st.tabs(["📈 Phân phối điểm", "📊 So sánh nhóm", "📄 Bảng & Tương quan"])
    with tab1:
        cL, cR = st.columns(2)
        with cL:
            fig = px.histogram(df, x=score_field, nbins=30, marginal="box",
                               title=f"Histogram + Boxplot – {score_field}")
            st.plotly_chart(fig, use_container_width=True)
        with cR:
            fig2 = px.violin(df, x="Class", y=score_field, color="Gender", box=True, points="all",
                             title=f"Violin by Class & Gender – {score_field}")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        grp = df.groupby(["Class","Gender"], as_index=False)[["Math","Reading","Writing","Average"]].mean().round(1)
        fig3 = px.bar(grp, x="Class", y="Average", color="Gender", barmode="group",
                      title="Average Score by Class & Gender", text_auto=True)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.dataframe(df.sample(min(500, len(df))))
        corr = df[["Math","Reading","Writing","Average","Attendance"]].corr().round(2)
        fig4 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig4, use_container_width=True)
        download_button(df, filename="student_performance.csv")

    if show_help:
        st.info("Gợi ý: Phối hợp histogram/box/violin/bar/heatmap để kể câu chuyện dữ liệu giáo dục.")

# ========== DATASET: UPLOAD CSV ==========
else:
    st.subheader("📤 Upload your CSV")
    uploaded = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded:
        # Đọc CSV linh hoạt (thử ; và ,)
        raw = uploaded.getvalue().decode("utf-8", errors="ignore")
        try:
            df = pd.read_csv(StringIO(raw))
        except Exception:
            df = pd.read_csv(StringIO(raw), sep=";")
        st.success(f"Tải thành công: {uploaded.name} • {df.shape[0]} rows, {df.shape[1]} cols")
        st.dataframe(df.head(200))

        # Widget cấu hình
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not date_cols:
            # thử parse cột tên "date" phổ biến
            for c in df.columns:
                if c.lower() in ("date","time","datetime","ngay","thoi_gian"):
                    try:
                        df[c] = pd.to_datetime(df[c])
                        date_cols.append(c)
                    except Exception:
                        pass

        with st.sidebar:
            st.subheader("Chart Config")
            x_col = st.selectbox("X axis", options=df.columns, index=0)
            y_cols = st.multiselect("Y axis (numeric)", options=numeric_cols, default=numeric_cols[:1])
            color_col = st.selectbox("Color (optional)", options=["(none)"] + list(df.columns), index=0)
            chart_type = st.radio("Chart type", ["Line","Bar","Scatter"], horizontal=True)

        # Vẽ
        if y_cols:
            if chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_cols,
                              color=None if color_col=="(none)" else color_col,
                              title="Custom Line Chart")
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_cols,
                             color=None if color_col=="(none)" else color_col,
                             title="Custom Bar Chart", barmode="group")
            else:
                # scatter chỉ cho 1 y
                y1 = y_cols[0]
                fig = px.scatter(df, x=x_col, y=y1,
                                 color=None if color_col=="(none)" else color_col,
                                 title="Custom Scatter")
            st.plotly_chart(fig, use_container_width=True)

        # Thêm table + download
        st.subheader("📄 Data Table")
        st.dataframe(df.head(1000))
        download_button(df, filename="uploaded_data.csv")

        if show_help:
            st.info("Gợi ý: Chọn cột X/Y, thêm Color để nhóm. Hãy đặt tiêu đề, nhãn trục rõ ràng để dashboard sạch.")

    else:
        st.info("Hãy upload một file CSV để bắt đầu. Ví dụ: dữ liệu bán hàng, ghi nhận thời tiết, v.v.")

# ========== FOOTER ==========
st.markdown("---")
st.caption("✅ Yêu cầu đạt: ≥3 biểu đồ/bảng • ≥1 widget tương tác • Thiết kế sạch (tiêu đề, nhãn, chú thích).")
