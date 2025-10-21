# app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
from forecasting import render_forecasting_tab


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="Rental Analytics Dashboard", page_icon="🚗", layout="wide")
DATA_PATH = "merged_df_further_cleaned.xlsx"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_top_value(s: pd.Series, default="—"):
    if s is None:
        return default
    s = s.dropna()
    return default if s.empty else s.value_counts().idxmax()

def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"
        
import functools

def _no_grid(px_func):
    """Wrapper that removes gridlines from all px figures."""
    @functools.wraps(px_func)
    def wrapper(*args, **kwargs):
        fig = px_func(*args, **kwargs)
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    return wrapper

# patch the functions you use
px.line = _no_grid(px.line)
px.bar = _no_grid(px.bar)
px.histogram = _no_grid(px.histogram)
px.scatter = _no_grid(px.scatter)


    
def maybe_kpi_or_bar(series: pd.Series, title: str, label_name: str):
    """
    If a category count series contains one category -> show KPI
    else -> show bar chart.
    """
    s = series.dropna()
    if s.shape[0] <= 1:
        label = s.index[0] if s.shape[0] == 1 else "—"
        value = int(s.iloc[0]) if s.shape[0] == 1 else 0
        st.subheader(title)
        kpi_col, _ = st.columns([1, 3])
        with kpi_col:
            st.metric(label=f"{label_name} — {label}", value=fmt_int(value))
        return
    df_counts = s.rename_axis(label_name).reset_index(name="count")
    fig = px.bar(df_counts, x="count", y=label_name, orientation="h", title=title)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def compute_time_bins(df: pd.DataFrame):
    """Build 3-hour checkout time bins."""
    if "Checkout Time" not in df.columns:
        return None
    ct = df["Checkout Time"]

    # Robust hour extraction: handles 'HH:MM[:SS]' or 'HHMM' or python time
    if pd.api.types.is_datetime64_any_dtype(ct):
        hours = pd.to_datetime(ct, errors="coerce").dt.hour
    else:
        try:
            # Try parsing as time already (datetime.time objects)
            hours = pd.to_datetime(ct.astype(str), errors="coerce").dt.hour
        except Exception:
            hours = pd.to_datetime(ct.astype(str).str.zfill(4), format="%H%M", errors="coerce").dt.hour

    bins = list(range(0, 25, 3))  # 0..24 by 3
    labels = [f"{i:02d}:00-{i+3:02d}:00" for i in bins[:-1]]
    labels[-1] = "21:00-24:00"
    cats = pd.cut(hours, bins=bins, right=False, include_lowest=True, labels=labels)
    cats = pd.Categorical(cats, categories=labels, ordered=True)
    s = pd.value_counts(cats, sort=False, dropna=False).reindex(labels).fillna(0).astype(int)
    return pd.DataFrame({"Time Bin": labels, "Count": s.values})

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)

    # Dates
    if "Checkout Date" in df.columns:
        df["Checkout Date"] = pd.to_datetime(df["Checkout Date"], errors="coerce")
    if "Checkin Date" in df.columns:
        df["Checkin Date"] = pd.to_datetime(df["Checkin Date"], errors="coerce")

    df["__date_idx__"] = df["Checkout Date"]
    df["row_id_for_counts"] = range(1, len(df) + 1)

    # Location (prioritize the provided name)
    LOC_CANDS = [
        "Checkout Location Name",               # <- user specified
        "Checkout Location",
        "Checkout Location code",
        "Branch",
        "Location"
    ]
    loc_col = next((c for c in LOC_CANDS if c in df.columns), None)
    df["__location__"] = df[loc_col].fillna("Unknown") if loc_col else "Unknown"

    # Channel (Broker vs Direct via money signals)
    COMM = df.get("Commission Amount", pd.Series(np.nan, index=df.index))
    TRAV = df.get("Travel Agent Prepay Tour Voucher Amount", pd.Series(np.nan, index=df.index))
    USED = df.get("Used Tour Voucher Amount", pd.Series(np.nan, index=df.index))
    broker_mask = (pd.to_numeric(COMM, errors="coerce").fillna(0) > 0) | \
                  (pd.to_numeric(TRAV, errors="coerce").fillna(0) > 0) | \
                  (pd.to_numeric(USED, errors="coerce").fillna(0) > 0)
    df["cust_channel"] = np.where(broker_mask, "Broker", "Direct")

    # Region (Gulf vs Local vs Other)
    GCC = {"AE","SA","QA","KW","OM","BH"}
    country_col = next(
        (c for c in ["Address Country Code", "Responsible Country Code", "Responsible Billing Country"] if c in df.columns),
        None
    )
    def region_from_country(x):
        if pd.isna(x): return "Unknown"
        s = str(x).strip().upper()
        if s in GCC: return "Gulf"
        if s in {"LB", "LEBANON"}: return "Local"
        return "Other"
    df["cust_region"] = df[country_col].apply(region_from_country) if country_col else "Unknown"

    # Weekend flag & weekday name
    if "Checkout Date" in df.columns:
        df["checkout_is_weekend"] = df["Checkout Date"].dt.weekday >= 5
        df["checkout_weekday"] = df["Checkout Date"].dt.day_name()

    # Price/discount engineering
    amt = pd.to_numeric(df.get("Net Time&Dist Amount", pd.Series(np.nan, index=df.index)), errors="coerce")
    days = pd.to_numeric(df.get("Days Charged Count", pd.Series(np.nan, index=df.index)), errors="coerce")
    if "Rental Length Days" in df.columns or "Rental Length Hours" in df.columns:
        days_alt = pd.to_numeric(df.get("Rental Length Days", 0), errors="coerce").fillna(0)
        hours_alt = pd.to_numeric(df.get("Rental Length Hours", 0), errors="coerce").fillna(0)
        dur_days = (days_alt * 24 + hours_alt) / 24.0
        denom = days.replace(0, np.nan).fillna(dur_days.replace(0, np.nan))
    else:
        denom = days.replace(0, np.nan)

    df["base_price_per_day"] = (amt / denom) / 100.0

    disc_pct = pd.to_numeric(df.get("Discount %", pd.Series(np.nan, index=df.index)), errors="coerce")
    if disc_pct.notna().any() and disc_pct.abs().max() > 100:
        df["discount_rate"] = disc_pct / 10000.0
    else:
        df["discount_rate"] = disc_pct / 100.0

    # Seasonal / holiday flags
    if "__date_idx__" in df.columns:
        m = df["__date_idx__"].dt.month
        day = df["__date_idx__"].dt.day
        df["is_summer"] = m.isin([6,7,8])
        df["is_christmas_newyear"] = ((m==12) & (day>=15)) | ((m==1) & (day<=7))
        eid_ranges = [
            ("2019-06-03","2019-06-06"), ("2019-08-11","2019-08-14"),
            ("2020-05-24","2020-05-26"), ("2020-07-31","2020-08-03"),
            ("2021-05-13","2021-05-16"), ("2021-07-20","2021-07-23"),
            ("2022-05-02","2022-05-05"), ("2022-07-09","2022-07-12"),
            ("2023-04-20","2023-04-23"), ("2023-06-28","2023-07-01"),
            ("2024-04-09","2024-04-12"), ("2024-06-16","2024-06-19"),
        ]
        eid_mask = False
        for a,b in eid_ranges:
            eid_mask = eid_mask | ((df["__date_idx__"] >= pd.to_datetime(a)) & (df["__date_idx__"] <= pd.to_datetime(b)))
        df["is_eid"] = eid_mask

    return df

# ------------------------------------------------------------
# Load & Filter UI
# ------------------------------------------------------------
df = load_data()

MIN_DT = pd.to_datetime(df["__date_idx__"]).min()
MAX_DT = pd.to_datetime(df["__date_idx__"]).max()

def _init_state():
    if "flt_date" not in st.session_state:
        st.session_state.flt_date = [MIN_DT.date(), MAX_DT.date()]
    for key in ["flt_loc", "flt_channel", "flt_region", "flt_vehicle"]:
        if key not in st.session_state:
            st.session_state[key] = []

def _reset_filters():
    st.session_state.flt_date = [MIN_DT.date(), MAX_DT.date()]
    st.session_state.flt_loc = []
    st.session_state.flt_channel = []
    st.session_state.flt_region = []
    st.session_state.flt_vehicle = []

_init_state()

st.title("🚗 Rental Analytics Dashboard")

with st.container():
    left, mid, right = st.columns([6, 3, 1])

    with left:
        st.subheader("Filters")
        c1, c2, c3 = st.columns([2, 2, 2])

        with c1:
            st.date_input(
                "Date range (Checkout Date)",
                value=st.session_state.flt_date,
                min_value=MIN_DT.date(),
                max_value=MAX_DT.date(),
                key="flt_date",
            )

        with c2:
            st.multiselect(
                "Locations",
                options=sorted(df["__location__"].dropna().unique()),
                default=st.session_state.flt_loc,
                key="flt_loc",
            )
            st.multiselect(
                "Channel",
                options=sorted(df["cust_channel"].dropna().unique()),
                default=st.session_state.flt_channel,
                key="flt_channel",
            )

        with c3:
            st.multiselect(
                "Region",
                options=sorted(df["cust_region"].dropna().unique()),
                default=st.session_state.flt_region,
                key="flt_region",
            )
            st.multiselect(
                "Vehicle Group",
                options=sorted(df.get("Vehicle Group Rented", pd.Series(dtype=object)).dropna().unique()),
                default=st.session_state.flt_vehicle,
                key="flt_vehicle",
            )

    with right:
        st.write("")
        st.write("")
        st.button("🔄 Reset filters", use_container_width=True, on_click=_reset_filters)

# Apply filters
date_start, date_end = st.session_state.flt_date
mask = df["__date_idx__"].between(pd.to_datetime(date_start), pd.to_datetime(date_end))
if st.session_state.flt_loc:
    mask &= df["__location__"].isin(st.session_state.flt_loc)
if st.session_state.flt_channel:
    mask &= df["cust_channel"].isin(st.session_state.flt_channel)
if st.session_state.flt_region:
    mask &= df["cust_region"].isin(st.session_state.flt_region)
if st.session_state.flt_vehicle and "Vehicle Group Rented" in df.columns:
    mask &= df["Vehicle Group Rented"].isin(st.session_state.flt_vehicle)

df_filtered = df.loc[mask].copy()
if df_filtered.empty:
    st.info("No rows match the current filters.")
    st.stop()

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Total Rentals", fmt_int(len(df_filtered)))

total_rev = pd.to_numeric(df_filtered.get("Net Time&Dist Amount", pd.Series(dtype=float)), errors="coerce").sum() / 100
k2.metric("Total Revenue", f"${total_rev:,.0f}")

if {"Days Charged Count", "Net Time&Dist Amount"}.issubset(df_filtered.columns):
    adr = (
        pd.to_numeric(df_filtered["Net Time&Dist Amount"], errors="coerce") /
        pd.to_numeric(df_filtered["Days Charged Count"], errors="coerce").replace(0, np.nan)
    ).mean() / 100
    k3.metric("Avg Daily Rate", f"${adr:,.0f}")
else:
    k3.metric("Avg Daily Rate", "—")

k4, k5, k6 = st.columns(3)
wkend_share = (df_filtered["Checkout Date"].dt.weekday >= 5).mean() * 100
k4.metric("Weekend Share", f"{wkend_share:,.1f}%")
k5.metric("Top Vehicle", safe_top_value(df_filtered.get("Vehicle Group Rented")))
k6.metric("Top Location", safe_top_value(df_filtered.get("__location__")))

st.markdown("---")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_dem, tab_cust, tab_fleet, tab_time, tab_price, tab_ops, tab_fc = st.tabs(
    ["📈 Demand & Seasonality", "👥 Customer Mix", "🚗 Fleet Mix",
     "⏱️ Time Patterns", "💵 Price & Discount", "🛠️ Operations", "🔮 Forecasting"]
)

# ---------------------- Demand & Seasonality ----------------------
with tab_dem:
    monthly = (
        df_filtered.dropna(subset=["__date_idx__"])
        .set_index("__date_idx__")
        .resample("M")["row_id_for_counts"].count()
        .rename("rentals").reset_index()
    )
    fig = px.line(monthly, x="__date_idx__", y="rentals", title="Monthly Rentals")
    fig.update_layout(xaxis_title="Date", yaxis_title="rentals")
    st.plotly_chart(fig, use_container_width=True)

    # Rentals per Year
    yearly = (
        df_filtered.assign(year=df_filtered["__date_idx__"].dt.year)
        .groupby("year", dropna=False)["row_id_for_counts"].count()
        .reset_index(name="rentals")
    )
    fig = px.bar(yearly, x="year", y="rentals", title="Rentals per Year")
    fig.update_layout(xaxis_title="Year", yaxis_title="Rentals")
    st.plotly_chart(fig, use_container_width=True)

    # Seasonality by Month
    seasonality = (
        df_filtered.assign(month=df_filtered["__date_idx__"].dt.month)
        .groupby("month", dropna=False)["row_id_for_counts"].count()
        .reset_index(name="rentals")
    )
    fig = px.bar(seasonality, x="month", y="rentals", title="Seasonality by Month")
    fig.update_layout(xaxis_title="Month", yaxis_title="Rentals")
    st.plotly_chart(fig, use_container_width=True)

    # Holiday / seasonal impact shares
    def share(mask):
        return float(mask.mean()) if len(mask) else np.nan
    holiday_summary = pd.DataFrame({
        "Period": ["share_eid", "share_christmas_newyear", "share_summer"],
        "share": [
            share(df_filtered["is_eid"]),
            share(df_filtered["is_christmas_newyear"]),
            share(df_filtered["is_summer"]),
        ]
    })
    fig = px.bar(holiday_summary,x="share",y="Period",orientation="h",title="Share of Rentals During Holiday/Seasonal Periods")
    fig.update_layout(xaxis_title="Share of Rentals",yaxis_title="Period")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Customer Mix ----------------------
with tab_cust:
    maybe_kpi_or_bar(df_filtered["cust_channel"].value_counts(), "Channel Breakdown", "channel")

    maybe_kpi_or_bar(df_filtered["cust_region"].value_counts(), "Region Breakdown", "region")

    maybe_kpi_or_bar(df_filtered["__location__"].value_counts().head(10), "Top Locations", "location")

# ---------------------- Fleet Mix ----------------------
with tab_fleet:
    if "Vehicle Group Rented" in df_filtered.columns:
        maybe_kpi_or_bar(
            df_filtered["Vehicle Group Rented"].value_counts().head(10),
            "Vehicle Group Distribution",
            "vehicle_group"
        )
    else:
        st.info("No 'Vehicle Group Rented' column found.")

# ---------------------- Time Patterns ----------------------
with tab_time:
    wd = (
        df_filtered["Checkout Date"].dt.day_name().value_counts()
        .rename_axis("weekday").reset_index(name="count")
    )
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wd["weekday"] = pd.Categorical(wd["weekday"], categories=order, ordered=True)
    wd = wd.sort_values("weekday")
    maybe_kpi_or_bar(wd.set_index("weekday")["count"], "Rentals by Weekday", "weekday")

    wk = df_filtered["checkout_is_weekend"].map({True:"Weekend", False:"Weekday"}).value_counts()
    maybe_kpi_or_bar(wk, "Rental Frequency: Weekend vs Weekday", "Is Weekend")

    tb = compute_time_bins(df_filtered)
    if tb is not None:
        fig = px.bar(tb, x="Time Bin", y="Count", title="Rental Frequency by Checkout Time (3-hour Bins)")
        fig.update_layout(xaxis_title="Time Bin", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No checkout time data.")

# ---------------------- Price & Discount ----------------------
with tab_price:
    # Monthly aggregates
    dfp = df_filtered.copy()
    dfp = dfp.dropna(subset=["__date_idx__"])
    monthly_agg = (
        dfp.set_index("__date_idx__")
        .resample("M")
        .agg(
            rentals=("row_id_for_counts", "count"),
            avg_base_price=("base_price_per_day", "mean"),
            avg_discount_rate=("discount_rate", "mean"),
        )
        .reset_index()
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.bar(monthly_agg, x="__date_idx__", y="rentals", title="Monthly Rentals")
        fig.update_layout(xaxis_title="Date", yaxis_title="rentals")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(monthly_agg, x="__date_idx__", y="avg_base_price",
                     title="Average Base Price per Day (Monthly)")
        fig.update_layout(xaxis_title="Date", yaxis_title="avg_base_price")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.bar(monthly_agg, x="__date_idx__", y="avg_discount_rate",
                     title="Average Discount Rate (Monthly)")
        fig.update_layout(xaxis_title="Date", yaxis_title="avg_discount_rate")
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        if df_filtered["base_price_per_day"].notna().any():
            fig = px.histogram(dfp, x="base_price_per_day", title="Distribution of Base Price per Day")
            fig.update_layout(xaxis_title="Base Price per Day", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No base price data.")
    with c5:
        if df_filtered["discount_rate"].notna().any():
            fig = px.histogram(dfp, x="discount_rate", title="Distribution of Discount Rate")
            fig.update_layout(xaxis_title="Discount Rate", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No discount rate data.")

with tab_fc:
    # pass the filtered dataframe so forecasts reflect the active filters
    render_forecasting_tab(df_filtered)
# ---------------------- Operations ----------------------
with tab_ops:
    c1, c2, c3 = st.columns(3)
    if "Rental Length Days" in df_filtered.columns:
        with c1:
            fig = px.histogram(df_filtered, x="Rental Length Days", title="Distribution of Rental Length (Days)")
            fig.update_layout(xaxis_title="Rental Length (Days)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
    if "Rental Length Hours" in df_filtered.columns:
        with c2:
            fig = px.histogram(df_filtered, x="Rental Length Hours", title="Distribution of Rental Length (Hours)")
            fig.update_layout(xaxis_title="Rental Length (Hours)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
    if "Days Charged Count" in df_filtered.columns:
        with c3:
            fig = px.histogram(df_filtered, x="Days Charged Count", title="Distribution of Days Charged")
            fig.update_layout(xaxis_title="Days Charged", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
