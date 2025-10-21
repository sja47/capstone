# forecasting.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import sqrt
from typing import Optional, Tuple, Dict, List

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LinearRegression

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import streamlit as st

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
H_DEFAULT = 12  # months

def _to_monthly_count(df: pd.DataFrame, date_col="__date_idx__", id_col="row_id_for_counts") -> pd.Series:
    d = df.dropna(subset=[date_col]).set_index(date_col)
    if id_col not in d.columns:
        d[id_col] = range(1, len(d) + 1)
    y = d[id_col].resample("M").count().asfreq("M").fillna(0)
    return y

def _to_monthly_sum(df: pd.DataFrame, value_col: str, date_col="__date_idx__") -> pd.Series:
    d = df.dropna(subset=[date_col]).set_index(date_col)
    y = d[value_col].resample("M").sum().asfreq("M").fillna(0)
    return y

def _ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _build_exog(df: pd.DataFrame, date_col="__date_idx__") -> pd.DataFrame:
    # checkout_month + Daily Rate (if missing, create it)
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d["checkout_month"] = d[date_col].dt.month

    if "Daily Rate" not in d.columns:
        days = _ensure_numeric(d.get("Days Charged Count", pd.Series(index=d.index))).replace(0, 1)
        net  = _ensure_numeric(d.get("Net Time&Dist Amount", pd.Series(index=d.index)))
        d["Daily Rate"] = (net / days) / 100.0  # to currency
        d["Daily Rate"] = d["Daily Rate"].fillna(d["Daily Rate"].median())

    exog = (
        d.set_index(date_col)[["checkout_month", "Daily Rate"]]
         .apply(_ensure_numeric)
         .resample("M").mean()
         .asfreq("M")
    )
    exog = exog.fillna(method="ffill").fillna(method="bfill").fillna(0)
    return exog

def _exog_future_repeat_last_year(exog_hist: pd.DataFrame, H: int) -> pd.DataFrame:
    idx = pd.date_range(start=exog_hist.index[-1] + pd.offsets.MonthEnd(1), periods=H, freq="M")
    fut = pd.DataFrame(index=idx, columns=exog_hist.columns)
    if len(exog_hist) >= 12:
        last_year = exog_hist.iloc[-12:]
        by_month = last_year.groupby(last_year.index.month).mean()
        for d in idx:
            fut.loc[d] = by_month.loc[d.month].values
    else:
        fut[:] = exog_hist.iloc[-1].values
    return fut.astype(float)

def _create_features(y: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=y.index)
    X["month"] = y.index.month
    X["year"]  = y.index.year
    for k in range(1, 13):
        X[f"lag_{k}"] = y.shift(k)
    return X

def _rmse(a: pd.Series, b: pd.Series) -> float:
    return float(sqrt(mean_squared_error(a, b)))

# -----------------------------------------------------------------------------
# Model bake-offs (small grids to keep it fast but faithful to your code)
# -----------------------------------------------------------------------------
def _bakeoff_arima(y_tr: pd.Series, y_te: pd.Series) -> Tuple[str, float, str]:
    orders = [(p, d, q) for p in range(0, 3) for d in range(0, 3) for q in range(0, 3)]
    best = (np.inf, None)
    for o in orders:
        try:
            m = ARIMA(y_tr, order=o).fit()
            fc = m.forecast(steps=len(y_te))
            rmse = _rmse(y_te, fc)
            if rmse < best[0]: best = (rmse, o)
        except Exception:
            continue
    return ("ARIMA", best[0], f"{best[1]}")

def _bakeoff_sarima(y_tr: pd.Series, y_te: pd.Series) -> Tuple[str, float, str]:
    orders     = [(p, d, q) for p in range(0, 2) for d in range(0, 2) for q in range(0, 2)]
    seas_orders= [(P, D, Q, 12) for P in range(0, 2) for D in range(0, 2) for Q in range(0, 2)]
    best = (np.inf, None, None)
    for o in orders:
        for so in seas_orders:
            try:
                m = SARIMAX(y_tr, order=o, seasonal_order=so,
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                fc = m.forecast(steps=len(y_te))
                rmse = _rmse(y_te, fc)
                if rmse < best[0]: best = (rmse, o, so)
            except Exception:
                continue
    return ("SARIMA", best[0], f"{best[1]} x {best[2]}")

def _bakeoff_arimax(y_tr, y_te, X_tr, X_te) -> Tuple[str, float, str]:
    orders = [(p, d, q) for p in range(0, 2) for d in range(0, 2) for q in range(0, 2)]
    best = (np.inf, None)
    for o in orders:
        try:
            m = ARIMA(y_tr, exog=X_tr, order=o).fit()
            fc = m.forecast(steps=len(y_te), exog=X_te)
            rmse = _rmse(y_te, fc)
            if rmse < best[0]: best = (rmse, o)
        except Exception:
            continue
    return ("ARIMAX", best[0], f"{best[1]}")

def _bakeoff_sarimax(y_tr, y_te, X_tr, X_te) -> Tuple[str, float, str]:
    orders     = [(p, d, q) for p in range(0, 2) for d in range(0, 2) for q in range(0, 2)]
    seas_orders= [(P, D, Q, 12) for P in range(0, 2) for D in range(0, 2) for Q in range(0, 2)]
    best = (np.inf, None, None)
    for o in orders:
        for so in seas_orders:
            try:
                m = SARIMAX(y_tr, exog=X_tr, order=o, seasonal_order=so,
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                fc = m.forecast(steps=len(y_te), exog=X_te)
                rmse = _rmse(y_te, fc)
                if rmse < best[0]: best = (rmse, o, so)
            except Exception:
                continue
    return ("SARIMAX", best[0], f"{best[1]} x {best[2]}")

def _bakeoff_ml(y: pd.Series, H: int) -> Tuple[str, float, str]:
    X = _create_features(y)
    X = X.dropna()
    y2 = y[X.index]
    if len(X) < H + 12:
        return ("Linear Regression", np.inf, "N/A")

    X_tr, X_te = X[:-H], X[-H:]
    y_tr, y_te = y2[:-H], y2[-H:]

    models: Dict[str, object] = {
        "Random Forest": RandomForestRegressor(n_estimators=400, random_state=42),
        "Extra Trees":   ExtraTreesRegressor(n_estimators=400, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVR (RBF)": SVR(kernel="rbf", C=10.0, epsilon=0.2),
        "Ridge": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression(),
    }

    best = (np.inf, None)
    for name, mdl in models.items():
        try:
            mdl.fit(X_tr, y_tr)
            pred = mdl.predict(X_te)
            rmse = _rmse(y_te, pred)
            if rmse < best[0]:
                best = (rmse, name)
        except Exception:
            continue
    return (best[1], best[0], "N/A")

def _train_champion_and_forecast(
    y: pd.Series,
    H: int,
    champion_name: str,
    order_str: str,
    exog_hist: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None
) -> pd.Series:
    """Train champion on full history and return a length-H forecast (no confidence interval)."""
    if "SARIMA" in champion_name or "ARIMA" in champion_name:
        # parse "(p,d,q)" and maybe "(P,D,Q,s)"
        import re
        parts = re.findall(r"\(.*?\)", order_str)
        order = eval(parts[0]) if parts else (1, 1, 1)
        seas  = eval(parts[1]) if len(parts) > 1 else (0, 0, 0, 0)
        try:
            mdl = SARIMAX(
                y,
                exog=exog_hist if "X" in champion_name else None,
                order=order,
                seasonal_order=seas,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            fc = mdl.forecast(steps=H, exog=exog_future if "X" in champion_name else None)
            return fc.clip(lower=0)
        except Exception:
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthEnd(1), periods=H, freq="M")
            return pd.Series([y.iloc[-1]] * H, index=idx)
    else:
        # ML champion: iterative one-step ahead using lag features
        X_full = _create_features(y).dropna()
        if X_full.empty:
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthEnd(1), periods=H, freq="M")
            return pd.Series([y.iloc[-1]] * H, index=idx)

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=400, random_state=42),
            "Extra Trees":   ExtraTreesRegressor(n_estimators=400, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVR (RBF)": SVR(kernel="rbf", C=10.0, epsilon=0.2),
            "Ridge": Ridge(alpha=1.0),
            "Linear Regression": LinearRegression(),
        }
        mdl = models.get(champion_name, LinearRegression())
        y_aligned = y[X_full.index]
        mdl.fit(X_full, y_aligned)

        future_idx = pd.date_range(y.index[-1] + pd.offsets.MonthEnd(1), periods=H, freq="M")
        cur_feat = X_full.iloc[-1].copy()
        preds: List[float] = []

        for step in range(H):
            next_date = future_idx[step]
            cur_feat["month"] = next_date.month
            cur_feat["year"]  = next_date.year
            x_df = pd.DataFrame([cur_feat.values], columns=cur_feat.index)
            try:
                pred = float(mdl.predict(x_df)[0])
            except Exception:
                pred = float(y.iloc[-1])
            preds.append(max(0.0, pred))

            # shift lags
            for j in range(12, 1, -1):
                if f"lag_{j}" in cur_feat and f"lag_{j-1}" in cur_feat:
                    cur_feat[f"lag_{j}"] = cur_feat[f"lag_{j-1}"]
            if "lag_1" in cur_feat:
                cur_feat["lag_1"] = preds[-1]

        return pd.Series(preds, index=future_idx)

def _champion_summary(y: pd.Series, H: int,
                      exog: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Tuple[str,str,float]]:
    """Do the bake-off and return (summary_df, (model_name, order_str, rmse))."""
    # basic split: last H months as test (if not enough, use 80/20)
    if len(y) < max(24, H + 6):
        cut = int(len(y) * 0.8)
    else:
        cut = len(y) - H
    cut = max(12, min(len(y)-1, cut))

    y_tr, y_te = y[:cut], y[cut:]
    results = []

    # ARIMA / SARIMA
    try:
        name, rmse, order = _bakeoff_arima(y_tr, y_te);   results.append({"Model": name, "RMSE": rmse, "Order": order})
    except Exception: pass
    try:
        name, rmse, order = _bakeoff_sarima(y_tr, y_te);  results.append({"Model": name, "RMSE": rmse, "Order": order})
    except Exception: pass

    # With exog if provided
    if exog is not None and not exog.empty:
        X_tr, X_te = exog.loc[y_tr.index], exog.loc[y_te.index]
        try:
            name, rmse, order = _bakeoff_arimax(y_tr, y_te, X_tr, X_te)
            results.append({"Model": name, "RMSE": rmse, "Order": order})
        except Exception: pass
        try:
            name, rmse, order = _bakeoff_sarimax(y_tr, y_te, X_tr, X_te)
            results.append({"Model": name, "RMSE": rmse, "Order": order})
        except Exception: pass

    # ML pack
    try:
        name, rmse, order = _bakeoff_ml(y, H)
        results.append({"Model": name, "RMSE": rmse, "Order": order})
    except Exception:
        pass

    summary = pd.DataFrame(results).sort_values("RMSE", ascending=True, na_position="last").reset_index(drop=True)
    if summary.empty:
        # fallback "model"
        summary = pd.DataFrame([{"Model": "Linear Regression", "RMSE": np.inf, "Order": "N/A"}])

    best = summary.iloc[0]
    return summary, (str(best["Model"]), str(best["Order"]), float(best["RMSE"]) if np.isfinite(best["RMSE"]) else np.inf)

def _plot_history_forecast(y: pd.Series, fc: pd.Series, title: str, ylab: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index,  y=y, name="History", mode="lines"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc, name="Forecast", mode="lines", line=dict(dash="dot")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=ylab)
    return fig

# -----------------------------------------------------------------------------
# Public entry point for the tab
# -----------------------------------------------------------------------------
def render_forecasting_tab(df_filtered: pd.DataFrame):
    st.header("🔮 Forecasting")

    # Controls
    colA, colB = st.columns([1, 3])
    with colA:
        H = st.select_slider("Forecast horizon (months)", [6, 9, 12, 18, 24], value=12)
    with colB:
        st.caption("Champion model is chosen via a quick bake-off (ARIMA/SARIMA/ARIMAX/SARIMAX + a compact ML pack) on the filtered data.")

    if df_filtered.empty:
        st.info("No rows match the current filters.")
        return

    # ------------------ Plot 1: Total Rentals (Champion) ------------------
    y_rentals = _to_monthly_count(df_filtered)
    if len(y_rentals.dropna()) < 8:
        st.info("Not enough monthly points to forecast total rentals.")
    else:
        exog_r = _build_exog(df_filtered)
        exog_r_future = _exog_future_repeat_last_year(exog_r, H)
        _, (best_name, best_order, best_rmse) = _champion_summary(y_rentals, H, exog_r)

        fc_mean = _train_champion_and_forecast(
            y_rentals, H, best_name, best_order,
            exog_hist=exog_r if "X" in best_name else None,
            exog_future=exog_r_future if "X" in best_name else None
        )
        fig1 = _plot_history_forecast(
            y_rentals, fc_mean,
            f"Forecast for Total Rentals — {best_name} (RMSE: {best_rmse:,.2f})",
            "Number of Rentals"
        )
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # ------------------ Plot 2: Top 5 Vehicle Groups (each on its own plot) ------------------
    st.subheader("Top Vehicle Groups — Champion Forecasts")
    vg_col = "Vehicle Group Rented"
    if vg_col not in df_filtered.columns:
        st.info("Column 'Vehicle Group Rented' not found.")
    else:
        top5 = df_filtered[vg_col].dropna().value_counts().head(5).index.tolist()
        if not top5:
            st.info("No vehicle group data available after filters.")
        else:
            cols = st.columns(2)  # lay out 2 charts per row
            i = 0
            for cat in top5:
                df_cat = df_filtered[df_filtered[vg_col] == cat]
                y_cat = _to_monthly_count(df_cat)
                if len(y_cat.dropna()) < 8 or y_cat.sum() == 0:
                    continue

                exog_c = _build_exog(df_cat)
                exog_c_future = _exog_future_repeat_last_year(exog_c, H)
                _, (best_name_c, best_order_c, best_rmse_c) = _champion_summary(y_cat, H, exog_c)

                fc_mean_c = _train_champion_and_forecast(
                    y_cat, H, best_name_c, best_order_c,
                    exog_hist=exog_c if "X" in best_name_c else None,
                    exog_future=exog_c_future if "X" in best_name_c else None
                )

                fig_cat = _plot_history_forecast(
                    y_cat, fc_mean_c,
                    f"{cat} — {best_name_c} (RMSE: {best_rmse_c:,.2f})",
                    "Number of Rentals"
                )
                with cols[i % 2]:
                    st.plotly_chart(fig_cat, use_container_width=True)
                i += 1

    st.markdown("---")

    # ------------------ Plot 3: Total Revenue (Champion) ------------------
    if "Net Time&Dist Amount" not in df_filtered.columns:
        st.info("Column 'Net Time&Dist Amount' not found; revenue forecast skipped.")
        return

    # Revenue is stored in cents; scale to currency
    y_rev = _to_monthly_sum(df_filtered, "Net Time&Dist Amount") / 100.0
    if len(y_rev.dropna()) < 8:
        st.info("Not enough monthly points to forecast revenue.")
        return

    exog_rev = _build_exog(df_filtered)
    exog_rev_future = _exog_future_repeat_last_year(exog_rev, H)
    _, (best_name_r, best_order_r, best_rmse_r) = _champion_summary(y_rev, H, exog_rev)

    fc_mean_r = _train_champion_and_forecast(
        y_rev, H, best_name_r, best_order_r,
        exog_hist=exog_rev if "X" in best_name_r else None,
        exog_future=exog_rev_future if "X" in best_name_r else None
    )

    fig3 = _plot_history_forecast(
        y_rev, fc_mean_r,
        f"Forecast for Total Revenue — {best_name_r} (RMSE: {best_rmse_r:,.2f})",
        "Total Revenue"
    )
    st.plotly_chart(fig3, use_container_width=True)
