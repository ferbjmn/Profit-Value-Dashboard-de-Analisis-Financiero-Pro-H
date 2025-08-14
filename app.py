# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO
#      • ROIC y WACC (Kd y tasa efectiva por empresa)
#      • Resumen agrupado y ORDENADO automáticamente por Sector
#      • Gráficos por SECTOR (con lotes de 10 para deuda y crecimiento)
#      • Ejes Y AUTO-AJUSTADOS y SIN mínimos por sector
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import time

# -------------------------------------------------------------
# ⚙️ Configuración global
# -------------------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Parámetros CAPM por defecto (editables)
# -------------------------------------------------------------
Rf  = 0.0435   # riesgo libre
Rm  = 0.085    # retorno mercado
Tc0 = 0.21     # tasa impositiva por defecto

# Orden de sectores
SECTOR_RANK = {
    "Consumer Defensive": 1,
    "Consumer Cyclical": 2,
    "Healthcare": 3,
    "Technology": 4,
    "Financial Services": 5,
    "Industrials": 6,
    "Communication Services": 7,
    "Energy": 8,
    "Real Estate": 9,
    "Utilities": 10,
    "Basic Materials": 11,
    "Unknown": 99,
}

MAX_TICKERS_PER_CHART = 10  # lotes para Deuda/Liquidez y Crecimiento

# =============================================================
# 1) FUNCIONES AUXILIARES
# =============================================================
def safe_first(obj):
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
    return obj.iloc[0] if hasattr(obj, "iloc") and not obj.empty else obj

def seek_row(df, keys):
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series([0], index=df.columns[:1])

def calc_ke(beta):
    return Rf + beta * (Rm - Rf)

def calc_kd(interest, debt):
    return interest / debt if debt else 0

def calc_wacc(mcap, debt, ke, kd, t):
    total = (mcap or 0) + (debt or 0)
    return (mcap/total)*ke + (debt/total)*kd*(1-t) if total else None

def cagr4(fin, metric):
    if metric not in fin.index:
        return None
    v = fin.loc[metric].dropna().iloc[:4]
    return (v.iloc[0]/v.iloc[-1])**(1/(len(v)-1))-1 if len(v)>1 and v.iloc[-1] else None

def chunk_df(df, size=MAX_TICKERS_PER_CHART):
    if df.empty:
        return []
    return [df.iloc[i:i+size] for i in range(0, len(df), size)]

def sector_sorted(df):
    df["Sector"] = df["Sector"].fillna("Unknown")
    df["SectorRank"] = df["Sector"].map(SECTOR_RANK).fillna(999).astype(int)
    return df.sort_values(["SectorRank", "Sector", "Ticker"]).reset_index(drop=True)

def auto_ylim(ax, values, pad=0.10):
    """Ajuste automático del eje Y."""
    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype="float64")
    else:
        arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax == vmin:
        ymin = vmin - abs(vmin)*pad - 1
        ymax = vmax + abs(vmax)*pad + 1
        ax.set_ylim(ymin, ymax)
        return
    if vmin >= 0:
        ymin = 0
        ymax = vmax * (1 + pad)
    elif vmax <= 0:
        ymax = 0
        ymin = vmin * (1 + pad)
    else:
        m = max(abs(vmin), abs(vmax)) * (1 + pad)
        ymin, ymax = -m, m
    ax.set_ylim(ymin, ymax)

# =============================================================
# 2) OBTENER DATOS POR EMPRESA
# =============================================================
def obtener_datos_financieros(tk, Tc_def):
    tkr  = yf.Ticker(tk)
    info = tkr.info
    bs   = tkr.balance_sheet
    fin  = tkr.financials
    cf   = tkr.cashflow
    if not info or bs.empty:
        raise ValueError("Sin datos de balance/info")

    beta  = info.get("beta", 1)
    ke    = calc_ke(beta)

    debt  = safe_first(seek_row(bs, ["Total Debt", "Long Term Debt"])) or info.get("totalDebt", 0)
    cash  = safe_first(seek_row(bs, [
        "Cash And Cash Equivalents",
        "Cash And Cash Equivalents At Carrying Value",
        "Cash Cash Equivalents And Short Term Investments",
    ]))
    equity= safe_first(seek_row(bs, ["Common Stock Equity", "Total Stockholder Equity"]))

    interest = safe_first(seek_row(fin, ["Interest Expense"]))
    ebt      = safe_first(seek_row(fin, ["Ebt", "EBT"]))
    tax_exp  = safe_first(seek_row(fin, ["Income Tax Expense"]))
    ebit     = safe_first(seek_row(fin, ["EBIT", "Operating Income",
                                         "Earnings Before Interest and Taxes"]))

    kd   = calc_kd(interest, debt)
    tax  = tax_exp / ebt if ebt else Tc_def
    mcap = info.get("marketCap", 0)
    wacc = calc_wacc(mcap, debt, ke, kd, tax)

    nopat = ebit * (1 - tax) if ebit is not None else None
    invested = (equity or 0) + ((debt or 0) - (cash or 0))
    roic = nopat / invested if (nopat is not None and invested) else None
    eva  = (roic - wacc) * invested if all(v is not None for v in (roic, wacc, invested)) else None

    price = info.get("currentPrice")
    fcf   = safe_first(seek_row(cf, ["Free Cash Flow"]))
    shares= info.get("sharesOutstanding")
    pfcf  = price / (fcf/shares) if (fcf and shares) else None

    # <<< claves robustas para evitar None y que no se caigan del DataFrame >>>
    nombre    = info.get("longName") or info.get("shortName") or info.get("displayName") or tk
    pais      = info.get("country")  or info.get("countryCode") or "N/D"
    industria = info.get("industry") or info.get("industryKey") or info.get("industryDisp") or "N/D"

    return {
        "Ticker": tk,
        "Nombre": nombre,
        "País": pais,
        "Industria": industria,
        "Sector": info.get("sector", "Unknown"),
        "Precio": price,
        "P/E": info.get("trailingPE"),
        "P/B": info.get("priceToBook"),
        "P/FCF": pfcf,
        "Dividend Yield %": info.get("dividendYield"),
        "Payout Ratio": info.get("payoutRatio"),
        "ROA": info.get("returnOnAssets"),
        "ROE": info.get("returnOnEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Quick Ratio": info.get("quickRatio"),
        "Debt/Eq": info.get("debtToEquity"),
        "LtDebt/Eq": info.get("longTermDebtToEquity"),
        "Oper Margin": info.get("operatingMargins"),
        "Profit Margin": info.get("profitMargins"),
        "WACC": wacc,
        "ROIC": roic,
        "EVA": eva,
        "Revenue Growth": cagr4(fin, "Total Revenue"),
        "EPS Growth":     cagr4(fin, "Net Income"),
        "FCF Growth":     cagr4(cf, "Free Cash Flow") or cagr4(cf, "Operating Cash Flow"),
        "MarketCap":      mcap,
    }

# =============================================================
# 3) INTERFAZ PRINCIPAL
# =============================================================
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("⚙️ Configuración")
        t_in = st.text_area("Tickers (coma)",
                            "HRL, AAPL, MSFT, ABT, O, XOM, KO, JNJ, CLX, CHD, CB, DDOG")
        max_t = st.slider("Máx tickers", 1, 100, 30)
        st.markdown("---")
        global Rf, Rm, Tc0
        Rf  = st.number_input("Risk-free (%)", 0.0, 20.0, 4.35)/100
        Rm  = st.number_input("Market return (%)", 0.0, 30.0, 8.5)/100
        Tc0 = st.number_input("Tax rate default (%)", 0.0, 50.0, 21.0)/100

    tickers = [t.strip().upper() for t in t_in.split(",") if t.strip()][:max_t]

    # ---------- Botón Analizar ----------
    if st.button("🔍 Analizar", type="primary"):
        if not tickers:
            st.warning("Ingresa al menos un ticker")
            return

        datos, errs, bar = [], [], st.progress(0)
        for i, tk in enumerate(tickers, 1):
            try:
                datos.append(obtener_datos_financieros(tk, Tc0))
            except Exception as e:
                errs.append({"Ticker": tk, "Error": str(e)})
            bar.progress(i / len(tickers))
            time.sleep(1)
        bar.empty()

        if not datos:
            st.error("Sin datos válidos.")
            if errs: st.table(pd.DataFrame(errs))
            return

        # df base y copia para display
        df = sector_sorted(pd.DataFrame(datos))
        df_disp = df.copy()

        # Formateo de porcentajes
        for col in ["Dividend Yield %", "Payout Ratio", "ROA", "ROE", "Oper Margin", "Profit Margin", "WACC", "ROIC"]:
            df_disp[col] = df_disp[col].apply(lambda x: f"{x*100:,.2f}%" if pd.notnull(x) else "N/D")

        # Asegurar que las nuevas columnas nunca sean NaN/None
        for c in ["Nombre", "País", "Industria"]:
            if c in df_disp.columns:
                df_disp[c] = df_disp[c].fillna("N/D").replace({None: "N/D", "": "N/D"})

        # =====================================================
        # Sección 1: Resumen General
        # =====================================================
        st.header("📋 Resumen General (agrupado por Sector)")
        resumen_cols = [
            "Ticker", "Nombre", "País", "Industria", "Sector",
            "Precio", "P/E", "P/B", "P/FCF",
            "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
            "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
            "WACC", "ROIC", "EVA"
        ]
        # Mostrar solo columnas existentes pero sin eliminar las 3 nuevas
        cols_show = [c for c in resumen_cols if c in df_disp.columns]
        st.dataframe(
            df_disp[cols_show],
            use_container_width=True,
            height=420
        )

        if errs:
            st.subheader("🚫 Tickers con error")
            st.table(pd.DataFrame(errs))

        sectors_ordered = df.sort_values("SectorRank")["Sector"].unique().tolist()

        # =====================================================
        # Sección 2: Análisis de Valoración
        # =====================================================
        st.header("💰 Análisis de Valoración (por Sector)")
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
            with st.expander(f"Sector: {sec}  •  {len(sec_df)} empresas", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 4))
                val = sec_df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                val.plot(kind="bar", ax=ax, rot=45)
                ax.set_ylabel("Ratio")
                auto_ylim(ax, val)
                st.pyplot(fig); plt.close()

        # Dividend Yield global
        st.subheader("Dividend Yield (%) • global")
        fig, ax = plt.subplots(figsize=(12, 4))
        dy = pd.DataFrame({"Dividend Yield %": (df["Dividend Yield %"]*100).values}, index=df["Ticker"])
        dy.plot(kind="bar", ax=ax, rot=45)
        ax.set_ylabel("%")
        auto_ylim(ax, dy)
        st.pyplot(fig); plt.close()

        # =====================================================
        # Sección 3: Rentabilidad y Eficiencia
        # =====================================================
        st.header("📈 Rentabilidad y Eficiencia")
        tabs = st.tabs(["ROE vs ROA (por sector)", "Márgenes (por sector)", "WACC vs ROIC (global)"])

        with tabs[0]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                with st.expander(f"Sector: {sec}  •  {len(sec_df)} empresas", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    rr = pd.DataFrame({
                        "ROE": (sec_df["ROE"]*100).values,
                        "ROA": (sec_df["ROA"]*100).values
                    }, index=sec_df["Ticker"])
                    rr.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, rr)
                    st.pyplot(fig); plt.close()

        with tabs[1]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                with st.expander(f"Sector: {sec}  •  {len(sec_df)} empresas", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mm = pd.DataFrame({
                        "Oper Margin": (sec_df["Oper Margin"]*100).values,
                        "Profit Margin": (sec_df["Profit Margin"]*100).values
                    }, index=sec_df["Ticker"])
                    mm.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, mm)
                    st.pyplot(fig); plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(12, 5))
            rw = pd.DataFrame({
                "ROIC": (df["ROIC"]*100).values,
                "WACC": (df["WACC"]*100).values
            }, index=df["Ticker"])
            rw.plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            ax.set_title("Creación de Valor: ROIC vs WACC")
            auto_ylim(ax, rw)
            st.pyplot(fig); plt.close()

        # =====================================================
        # Sección 4: Estructura de Capital y Liquidez
        # =====================================================
        st.header("🏦 Estructura de Capital y Liquidez (por sector)")
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
            with st.expander(f"Sector: {sec}  •  {len(sec_df)} empresas", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), start=1):
                    st.caption(f"Bloque {i}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Apalancamiento")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        lev = chunk[["Ticker", "Debt/Eq", "LtDebt/Eq"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                        lev.plot(kind="bar", stacked=True, ax=ax, rot=45)
                        ax.axhline(1, color="red", linestyle="--")
                        ax.set_ylabel("Ratio")
                        auto_ylim(ax, lev)
                        st.pyplot(fig); plt.close()
                    with c2:
                        st.caption("Liquidez")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        liq = chunk[["Ticker", "Current Ratio", "Quick Ratio"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                        liq.plot(kind="bar", ax=ax, rot=45)
                        ax.axhline(1, color="green", linestyle="--")
                        ax.set_ylabel("Ratio")
                        auto_ylim(ax, liq)
                        st.pyplot(fig); plt.close()

        # =====================================================
        # Sección 5: Crecimiento
        # =====================================================
        st.header("🚀 Crecimiento (CAGR 3-4 años, por sector)")
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
            with st.expander(f"Sector: {sec}  •  {len(sec_df)} empresas", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), start=1):
                    st.caption(f"Bloque {i}")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    gdf = pd.DataFrame({
                        "Revenue Growth": (chunk["Revenue Growth"]*100).values,
                        "EPS Growth":     (chunk["EPS Growth"]*100).values,
                        "FCF Growth":     (chunk["FCF Growth"]*100).values
                    }, index=chunk["Ticker"])
                    gdf.plot(kind="bar", ax=ax, rot=45)
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylabel("%")
                    auto_ylim(ax, gdf)
                    st.pyplot(fig); plt.close()

        # =====================================================
        # Sección 6: Análisis individual
        # =====================================================
        st.header("🔍 Análisis por Empresa")
        pick = st.selectbox("Selecciona empresa", df_disp["Ticker"].unique())
        det_disp = df_disp[df_disp["Ticker"] == pick].iloc[0]
        det_raw  = df[df["Ticker"] == pick].iloc[0]

        st.markdown(f"**{det_raw['Nombre']}**  \nPaís: {det_raw['País']}  \nIndustria: {det_raw['Industria']}")

        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Precio", f"${det_raw['Precio']:,.2f}" if det_raw['Precio'] else "N/D")
            st.metric("P/E", det_raw["P/E"])
            st.metric("P/B", det_raw["P/B"])
        with cB:
            st.metric("ROIC", det_disp["ROIC"])
            st.metric("WACC", det_disp["WACC"])
            st.metric("EVA", f"{det_raw['EVA']:,.0f}" if pd.notnull(det_raw["EVA"]) else "N/D")
        with cC:
            st.metric("ROE", det_disp["ROE"])
            st.metric("Dividend Yield", det_disp["Dividend Yield %"])
            st.metric("Debt/Eq", det_raw["Debt/Eq"])

        st.subheader("ROIC vs WACC")
        if pd.notnull(det_raw["ROIC"]) and pd.notnull(det_raw["WACC"]):
            fig, ax = plt.subplots(figsize=(5, 3.5))
            comp = pd.DataFrame({"ROIC": [det_raw["ROIC"]*100], "WACC": [det_raw["WACC"]*100]}, index=[pick])
            comp.plot(kind="bar", ax=ax, rot=0, legend=False, color=["green" if det_raw["ROIC"]>det_raw["WACC"] else "red", "gray"])
            ax.set_ylabel("%")
            auto_ylim(ax, comp)
            st.pyplot(fig); plt.close()
            st.success("✅ Crea valor (ROIC > WACC)" if det_raw["ROIC"] > det_raw["WACC"] else "❌ Destruye valor (ROIC < WACC)")
        else:
            st.info("Datos insuficientes para comparar ROIC/WACC")

# -------------------------------------------------------------
if __name__ == "__main__":
    main()
