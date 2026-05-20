import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid
import seaborn as sns
import matplotlib.pyplot as plt
import yahooquery as yq
from bs4 import BeautifulSoup
import requests
import re
import feedparser
import json
import os


# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

B3_DADOS_MERCADO_URL = (
    "https://sistemaswebb3-listados.b3.com.br/marketDataProxy/MarketDataCall/"
    "GetDownloadMarketData/RELATORIO_DADOS_DE_MERCADO.csv"
)
WATCHLIST_FILE = "data/watchlist.json"

COPOM_DATES = [
    ("2025-01-29", "2025-01-30"), ("2025-03-18", "2025-03-19"),
    ("2025-05-06", "2025-05-07"), ("2025-06-17", "2025-06-18"),
    ("2025-07-29", "2025-07-30"), ("2025-09-16", "2025-09-17"),
    ("2025-10-28", "2025-10-29"), ("2025-12-09", "2025-12-10"),
    ("2026-01-27", "2026-01-28"), ("2026-03-17", "2026-03-18"),
    ("2026-05-05", "2026-05-06"), ("2026-06-16", "2026-06-17"),
    ("2026-07-28", "2026-07-29"), ("2026-09-15", "2026-09-16"),
    ("2026-10-27", "2026-10-28"), ("2026-12-08", "2026-12-09"),
]

IBOV_COMPOSITION = {
    "VALE3": {"peso": 10.5, "setor": "Materiais Básicos"},
    "PETR4": {"peso": 8.2,  "setor": "Petróleo e Gás"},
    "ITUB4": {"peso": 7.8,  "setor": "Financeiro"},
    "BBDC4": {"peso": 5.3,  "setor": "Financeiro"},
    "PETR3": {"peso": 5.1,  "setor": "Petróleo e Gás"},
    "B3SA3": {"peso": 4.9,  "setor": "Financeiro"},
    "ABEV3": {"peso": 3.7,  "setor": "Consumo"},
    "BBAS3": {"peso": 3.5,  "setor": "Financeiro"},
    "WEGE3": {"peso": 3.2,  "setor": "Industrial"},
    "RENT3": {"peso": 2.9,  "setor": "Consumo"},
    "SUZB3": {"peso": 2.8,  "setor": "Materiais Básicos"},
    "ELET3": {"peso": 2.7,  "setor": "Utilidade Pública"},
    "GGBR4": {"peso": 2.6,  "setor": "Materiais Básicos"},
    "JBSS3": {"peso": 2.5,  "setor": "Consumo"},
    "RADL3": {"peso": 2.4,  "setor": "Consumo"},
    "LREN3": {"peso": 2.3,  "setor": "Consumo"},
    "HAPV3": {"peso": 2.2,  "setor": "Saúde"},
    "PRIO3": {"peso": 2.1,  "setor": "Petróleo e Gás"},
    "RAIL3": {"peso": 2.0,  "setor": "Industrial"},
    "CCRO3": {"peso": 1.9,  "setor": "Industrial"},
}


# ─────────────────────────────────────────────
# CSS / TEMA
# ─────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0f1923 100%);
        border-right: 1px solid #1e2d3d;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 12px 16px !important;
        transition: border-color .2s, background .2s;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(0,210,140,.35);
        background: rgba(0,210,140,.04);
    }

    /* Main gradient title */
    .main-title {
        background: linear-gradient(90deg, #00d28c, #00aaff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: .1rem;
    }
    .main-subtitle { color: #64748b; font-size: .9rem; margin-bottom: 1.2rem; }

    /* Section label */
    .sec-label {
        border-left: 3px solid #00d28c;
        padding-left: 10px;
        color: #cbd5e1;
        font-weight: 600;
        font-size: 1.05rem;
        margin: .8rem 0 .4rem;
    }

    /* Score colors */
    .score-hi { color: #22c55e; font-size: 2.2rem; font-weight: 800; }
    .score-md { color: #f59e0b; font-size: 2.2rem; font-weight: 800; }
    .score-lo { color: #ef4444; font-size: 2.2rem; font-weight: 800; }

    /* Tag badges */
    .tag-bull { background:rgba(34,197,94,.12); color:#22c55e;
                border:1px solid rgba(34,197,94,.3); border-radius:20px;
                padding:2px 10px; font-size:.78rem; }
    .tag-bear { background:rgba(239,68,68,.12); color:#ef4444;
                border:1px solid rgba(239,68,68,.3); border-radius:20px;
                padding:2px 10px; font-size:.78rem; }
    .tag-neut { background:rgba(148,163,184,.12); color:#94a3b8;
                border:1px solid rgba(148,163,184,.3); border-radius:20px;
                padding:2px 10px; font-size:.78rem; }

    /* Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border-color: rgba(255,255,255,.07) !important;
    }

    /* Expanders */
    [data-testid="stExpander"] details {
        border: 1px solid rgba(255,255,255,.07) !important;
        border-radius: 10px !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid rgba(0,210,140,.3);
        color: #00d28c;
        background: rgba(0,210,140,.07);
        transition: all .2s;
    }
    .stButton>button:hover {
        background: rgba(0,210,140,.15);
        border-color: #00d28c;
    }

    /* COPOM badge */
    .copom-next {
        background: linear-gradient(90deg, rgba(168,85,247,.15), rgba(0,170,255,.15));
        border: 1px solid rgba(168,85,247,.3);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 8px;
    }
    .copom-past { opacity: .45; }

    hr { border-color: rgba(255,255,255,.06) !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHED DATA FETCHERS
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def download_prices(tickers: tuple, start_date, end_date) -> pd.DataFrame:
    data = yf.download(list(tickers), start=start_date, end=end_date, progress=False)
    if data.empty:
        return pd.DataFrame()
    prices = data["Adj Close"] if "Adj Close" in data else data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        prices.columns = [list(tickers)[0].removesuffix(".SA")]
    else:
        prices.columns = [c.removesuffix(".SA") for c in prices.columns]
    ibov = yf.download("^BVSP", start=start_date, end=end_date, progress=False)
    prices["IBOV"] = ibov["Adj Close"] if "Adj Close" in ibov else ibov["Close"]
    return prices


@st.cache_data(ttl=300)
def fetch_ohlc(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    for col in required + (["Volume"] if "Volume" in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=required).sort_index()


@st.cache_data(ttl=3600)
def fetch_dividends(ticker: str) -> pd.Series:
    return yf.Ticker(ticker).dividends


@st.cache_data(ttl=300)
def download_rrg_prices(tickers_tuple: tuple, start_date, end_date) -> pd.DataFrame:
    data = yf.download(list(tickers_tuple), start=start_date, end=end_date, progress=False)
    if data.empty:
        return pd.DataFrame()
    prices = data["Adj Close"] if "Adj Close" in data else data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        prices.columns = [list(tickers_tuple)[0].removesuffix(".SA")]
    else:
        prices.columns = [c.removesuffix(".SA") for c in prices.columns]
    ibov = yf.download("^BVSP", start=start_date, end=end_date, progress=False)
    prices["IBOV"] = ibov["Adj Close"] if "Adj Close" in ibov else ibov["Close"]
    return prices


@st.cache_data(ttl=300)
def download_ibov_map_data(tickers_tuple: tuple) -> pd.DataFrame:
    return yf.download(list(tickers_tuple), period="2d", group_by="ticker", progress=False)


@st.cache_data(ttl=3600)
def fetch_quarterly_financials(ticker: str):
    t = yf.Ticker(ticker)
    return t.quarterly_financials, t.quarterly_income_stmt


@st.cache_data(ttl=3600)
def download_screener_data(tickers_tuple: tuple) -> pd.DataFrame:
    data = yf.download(list(tickers_tuple), period="1y", progress=False, auto_adjust=True)
    if data.empty:
        return pd.DataFrame()
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close.columns = [c.removesuffix(".SA") for c in close.columns]
    return close


@st.cache_data(ttl=3600)
def fetch_ticker_info(ticker: str) -> dict:
    """Retorna o dict .info do yfinance com todos os múltiplos do ativo."""
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def download_b3_dados_mercado_text() -> str:
    r = requests.get(B3_DADOS_MERCADO_URL, timeout=30)
    r.raise_for_status()
    return r.text


# ─────────────────────────────────────────────
# INDICADORES TÉCNICOS
# ─────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calc_bollinger(series: pd.Series, period=20, n_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + n_std * std, sma, sma - n_std * std


# ─────────────────────────────────────────────
# SCORE FUNDAMENTALISTA
# ─────────────────────────────────────────────

def calc_fund_score(pl, pvp, roe_pct, dy_pct, margin_pct) -> int:
    score = 0
    if pl and isinstance(pl, (int, float)) and 5 <= pl <= 20:
        score += 2
    if pvp and isinstance(pvp, (int, float)) and pvp < 3:
        score += 2
    if roe_pct and isinstance(roe_pct, (int, float)) and roe_pct > 15:
        score += 2
    if dy_pct and isinstance(dy_pct, (int, float)) and dy_pct > 3:
        score += 2
    if margin_pct and isinstance(margin_pct, (int, float)) and margin_pct > 10:
        score += 2
    return score


# ─────────────────────────────────────────────
# WATCHLIST
# ─────────────────────────────────────────────

def load_watchlist():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    return []


def save_watchlist(items):
    os.makedirs("data", exist_ok=True)
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(items, f)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def build_sidebar():
    st.image("images/avatar-renova-instagram.png")
    ticker_list = pd.read_csv("tickers/tickers_ibra.csv", index_col=0)

    tickers = st.multiselect("Selecione as Empresas", options=ticker_list, placeholder="Códigos")
    tickers = [t + ".SA" for t in tickers]
    start_date = st.date_input("De", value=datetime(2023, 1, 2), format="YYYY-MM-DD")
    end_date = st.date_input("Até", value=datetime.now().date(), format="YYYY-MM-DD")

    if not tickers:
        st.warning("Selecione pelo menos um ticker.")
        return None, None

    with st.spinner("Carregando dados..."):
        prices = download_prices(tuple(tickers), start_date, end_date)

    if prices.empty:
        st.error("Não foi possível obter dados. Verifique os códigos.")
        return None, None

    return tickers, prices


# ─────────────────────────────────────────────
# ABA: DASHBOARD
# ─────────────────────────────────────────────

def calculate_beta(returns, market_returns):
    cov = np.cov(returns, market_returns)[0][1]
    return cov / np.var(market_returns)


def main_dashboard(tickers, prices):
    weights = np.ones(len(tickers)) / len(tickers)
    prices["portfolio"] = prices.drop("IBOV", axis=1) @ weights
    norm_prices = 100 * prices / prices.iloc[0]
    returns = prices.pct_change()[1:]
    vols = returns.std() * np.sqrt(252)
    rets = (norm_prices.iloc[-1] - 100) / 100
    market_returns = returns["IBOV"]
    betas = {t: calculate_beta(returns[t], market_returns) for t in prices.columns if t != "IBOV"}

    mygrid = grid(3, 3, 3, 3, 3, 3, vertical_align="top")
    for t in prices.columns:
        c = mygrid.container(border=True)
        c.subheader(t, divider="red")
        colA, colB, colC, colD = c.columns([2, 6, 6, 6])
        if t == "portfolio":
            colA.image("images/pie-chart-dollar-svgrepo-com.svg", use_container_width=True)
        elif t == "IBOV":
            colA.image("images/pie-chart-svgrepo-com.svg", use_container_width=True)
        else:
            colA.image(f"https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{t}.png", width=85, use_container_width=True)
        colB.metric("Retorno", f"{rets[t]:.0%}")
        colC.metric("Volatilidade", f"{vols[t]:.0%}")
        colD.metric("Beta", f"{betas[t]:.2f}" if t in betas else "N/A")
        style_metric_cards(background_color="rgba(255,255,255,0)")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="sec-label">Desempenho Relativo</div>', unsafe_allow_html=True)
        st.line_chart(norm_prices, height=550)
    with col2:
        st.markdown('<div class="sec-label">Risco × Retorno</div>', unsafe_allow_html=True)
        fig = px.scatter(
            x=vols, y=rets, text=vols.index,
            color=rets / vols, color_continuous_scale="Blues",
            labels={"x": "Volatilidade", "y": "Retorno"},
            template="plotly_dark",
        )
        fig.update_traces(
            textfont_color="white",
            marker=dict(size=42, line=dict(width=1, color="DarkSlateGrey")),
            textfont_size=12,
        )
        fig.layout.height = 550
        fig.layout.coloraxis.colorbar.title = "Sharpe"
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# ABA: ANÁLISE TÉCNICA
# ─────────────────────────────────────────────

def technical_analysis_dashboard():
    st.markdown('<div class="sec-label">📈 Análise Técnica</div>', unsafe_allow_html=True)

    ticker_list = pd.read_csv("tickers/tickers_ibra.csv", index_col=0)
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    ticker       = c1.selectbox("Ativo", options=ticker_list)
    interval_lbl = c2.selectbox("Timeframe", ["Diário", "Semanal", "Mensal"])
    period       = c3.selectbox("Período", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
    show_volume  = c4.toggle("Volume", value=True)

    interval = {"Diário": "1d", "Semanal": "1wk", "Mensal": "1mo"}[interval_lbl]
    ticker_yf = ticker + ".SA"

    with st.spinner("Carregando candles..."):
        df = fetch_ohlc(ticker_yf, period=period, interval=interval)

    if df is None or df.empty:
        st.error("Não foi possível carregar dados para este ativo/período.")
        return

    # Cards de preço
    try:
        last_close  = float(df["Close"].iloc[-1])
        first_close = float(df["Close"].iloc[0])
        period_chg  = (last_close / first_close) - 1
        last_chg    = (last_close / float(df["Close"].iloc[-2]) - 1) if len(df) >= 2 else np.nan
        high52      = float(df["High"].max())
        low52       = float(df["Low"].min())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Preço (Close)",  f"R$ {last_close:,.2f}".replace(",","X").replace(".",",").replace("X","."))
        k2.metric(f"Var. período",  f"{period_chg:+.2%}")
        k3.metric(f"Var. candle",   f"{last_chg:+.2%}" if pd.notna(last_chg) else "—")
        k4.metric("Máx. período",   f"R$ {high52:,.2f}".replace(",","X").replace(".",",").replace("X","."))
        k5.metric("Mín. período",   f"R$ {low52:,.2f}".replace(",","X").replace(".",",").replace("X","."))
        style_metric_cards(background_color="rgba(255,255,255,0)")
    except Exception as e:
        st.warning(f"Não foi possível calcular métricas: {e}")

    # Opções de indicadores
    st.markdown("**Overlays e indicadores**")
    oc1, oc2, oc3, oc4, oc5, oc6 = st.columns(6)
    show_sma20  = oc1.checkbox("SMA 20",  value=True)
    show_sma50  = oc2.checkbox("SMA 50",  value=True)
    show_sma200 = oc3.checkbox("SMA 200", value=False)
    show_bb     = oc4.checkbox("Bollinger", value=False)
    show_rsi    = oc5.checkbox("RSI", value=True)
    show_macd   = oc6.checkbox("MACD", value=False)

    close = df["Close"].squeeze()
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    bb_up, bb_mid, bb_lo = calc_bollinger(close)
    rsi_vals            = calc_rsi(close)
    macd_line, sig_line, histogram = calc_macd(close)

    # Montar subplots dinamicamente
    rows, heights = [1], [0.65]
    if show_volume:
        rows.append(len(rows) + 1); heights.append(0.12)
    if show_rsi:
        rows.append(len(rows) + 1); heights.append(0.12)
    if show_macd:
        rows.append(len(rows) + 1); heights.append(0.12)

    total = sum(heights)
    heights = [h / total for h in heights]

    fig = make_subplots(
        rows=len(rows), cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.02,
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index.to_pydatetime(),
        open=df["Open"].values, high=df["High"].values,
        low=df["Low"].values,  close=df["Close"].values,
        name=ticker, increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    if show_sma20:
        fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA 20",
                                 line=dict(color="#f59e0b", width=1.2)), row=1, col=1)
    if show_sma50:
        fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50",
                                 line=dict(color="#60a5fa", width=1.2)), row=1, col=1)
    if show_sma200:
        fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA 200",
                                 line=dict(color="#e879f9", width=1.2)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=bb_up,  name="BB sup",
                                 line=dict(color="#94a3b8", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_lo,  name="BB inf",
                                 line=dict(color="#94a3b8", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(148,163,184,.06)"), row=1, col=1)

    cur_row = 2
    if show_volume and "Volume" in df.columns:
        colors = ["#22c55e" if c >= o else "#ef4444"
                  for c, o in zip(df["Close"].values, df["Open"].values)]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color=colors, opacity=0.6), row=cur_row, col=1)
        fig.update_yaxes(title_text="Volume", row=cur_row, col=1)
        cur_row += 1

    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=rsi_vals, name="RSI",
                                 line=dict(color="#a78bfa", width=1.5)), row=cur_row, col=1)
        fig.add_hline(y=70, line=dict(color="red",   width=.8, dash="dot"), row=cur_row, col=1)
        fig.add_hline(y=30, line=dict(color="green", width=.8, dash="dot"), row=cur_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=cur_row, col=1)
        cur_row += 1

    if show_macd:
        colors_hist = ["#22c55e" if v >= 0 else "#ef4444" for v in histogram.values]
        fig.add_trace(go.Bar(x=df.index, y=histogram, name="Histograma",
                             marker_color=colors_hist, opacity=0.7), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD",
                                 line=dict(color="#38bdf8", width=1.5)), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=sig_line, name="Sinal",
                                 line=dict(color="#f97316", width=1.5)), row=cur_row, col=1)
        fig.update_yaxes(title_text="MACD", row=cur_row, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        title=f"{ticker}  •  {interval_lbl}  •  {period}",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# ABA: SCREENER
# ─────────────────────────────────────────────

def screener_dashboard():
    st.markdown('<div class="sec-label">🔍 Screener de Ações</div>', unsafe_allow_html=True)
    st.caption("Filtros automáticos sobre os ativos do IBRA com base em indicadores técnicos e preço.")

    ticker_list = pd.read_csv("tickers/tickers_ibra.csv", index_col=0)
    all_tickers = [t + ".SA" for t in ticker_list.index.tolist()]

    col1, col2 = st.columns([3, 1])
    with col1:
        num = st.slider("Quantos ativos analisar (ordenados pela lista IBRA)", 10, min(80, len(all_tickers)), 40, step=10)
    with col2:
        st.write("")
        st.write("")
        run = st.button("▶ Rodar Screener", use_container_width=True)

    if not run:
        st.info("Configure os parâmetros e clique em **Rodar Screener**.")
        return

    subset = tuple(all_tickers[:num])
    with st.spinner(f"Baixando dados de {num} ativos (pode levar ~20 s)..."):
        close_df = download_screener_data(subset)

    if close_df.empty:
        st.error("Não foi possível baixar dados do screener.")
        return

    results = []
    for ticker in close_df.columns:
        s = close_df[ticker].dropna()
        if len(s) < 50:
            continue

        last   = s.iloc[-1]
        sma20  = s.rolling(20).mean().iloc[-1]
        sma50  = s.rolling(50).mean().iloc[-1]
        sma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan
        high52 = s.rolling(min(252, len(s))).max().iloc[-1]
        low52  = s.rolling(min(252, len(s))).min().iloc[-1]
        rsi    = calc_rsi(s).iloc[-1]

        # Tendência
        if not np.isnan(sma200) and last > sma50 > sma200:
            trend = "📈 Alta"
        elif not np.isnan(sma200) and last < sma50 < sma200:
            trend = "📉 Baixa"
        else:
            trend = "↔️ Lateral"

        # Sinal RSI
        if rsi < 30:
            rsi_signal = "🟢 Sobrevenda"
        elif rsi > 70:
            rsi_signal = "🔴 Sobrecompra"
        else:
            rsi_signal = "⚪ Neutro"

        results.append({
            "Ticker":         ticker,
            "Preço":          round(last, 2),
            "RSI":            round(rsi, 1),
            "Sinal RSI":      rsi_signal,
            "vs SMA20":       f"{(last/sma20-1)*100:+.1f}%",
            "vs SMA50":       f"{(last/sma50-1)*100:+.1f}%",
            "vs SMA200":      f"{(last/sma200-1)*100:+.1f}%" if not np.isnan(sma200) else "N/A",
            "Dist. Máx 52s":  f"{(last/high52-1)*100:+.1f}%",
            "Dist. Mín 52s":  f"{(last/low52-1)*100:+.1f}%",
            "Tendência":      trend,
        })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        st.warning("Nenhum dado processado.")
        return

    # Filtros rápidos
    st.markdown("**Filtros rápidos**")
    f1, f2, f3 = st.columns(3)
    filter_rsi    = f1.selectbox("Sinal RSI", ["Todos", "🟢 Sobrevenda", "🔴 Sobrecompra", "⚪ Neutro"])
    filter_trend  = f2.selectbox("Tendência", ["Todas", "📈 Alta", "📉 Baixa", "↔️ Lateral"])
    filter_sma200 = f3.checkbox("Acima da SMA200", value=False)

    if filter_rsi != "Todos":
        df_res = df_res[df_res["Sinal RSI"] == filter_rsi]
    if filter_trend != "Todas":
        df_res = df_res[df_res["Tendência"] == filter_trend]
    if filter_sma200:
        df_res = df_res[df_res["vs SMA200"].str.startswith("+")]

    st.markdown(f"**{len(df_res)} ativos** encontrados com os filtros selecionados.")
    st.dataframe(df_res.reset_index(drop=True), use_container_width=True, height=420)


# ─────────────────────────────────────────────
# ABA: MÚLTIPLOS
# ─────────────────────────────────────────────

def _fmt(value, suffix="", decimals=2, scale=1):
    """Formata um número ou retorna 'N/A' se inválido."""
    try:
        v = float(value) * scale
        if not np.isfinite(v):
            return "N/A"
        return f"{v:.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def multiples_dashboard(tickers):
    st.markdown('<div class="sec-label">📋 Múltiplos Financeiros</div>', unsafe_allow_html=True)

    if not tickers:
        st.warning("Selecione pelo menos um ticker.")
        return

    financial_data = []

    for ticker in tickers:
        tk = ticker.removesuffix(".SA")
        with st.spinner(f"Carregando dados de {tk}..."):
            info = fetch_ticker_info(ticker)

        if not info:
            st.warning(f"{tk}: dados não disponíveis no Yahoo Finance.")
            continue

        # Coleta dos campos — yfinance .info é a fonte mais confiável para B3
        pl        = info.get("trailingPE")
        pvp       = info.get("priceToBook")
        lpa       = info.get("trailingEps")
        roe_raw   = info.get("returnOnEquity")       # decimal ex: 0.18 = 18%
        dy_raw    = info.get("dividendYield")         # decimal ex: 0.06 = 6%
        margin    = info.get("profitMargins")         # decimal
        ebitda_m  = info.get("enterpriseToEbitda")
        preco     = info.get("currentPrice") or info.get("regularMarketPrice")
        mktcap    = info.get("marketCap")
        high52    = info.get("fiftyTwoWeekHigh")
        low52     = info.get("fiftyTwoWeekLow")
        nome      = info.get("longName") or info.get("shortName") or tk

        roe_pct    = roe_raw  * 100 if isinstance(roe_raw,  float) else None
        dy_pct     = dy_raw   * 100 if isinstance(dy_raw,   float) else None
        margin_pct = margin   * 100 if isinstance(margin,   float) else None
        score      = calc_fund_score(pl, pvp, roe_pct, dy_pct, margin_pct)

        financial_data.append({
            "Ticker":        tk,
            "Nome":          nome,
            "Preço":         _fmt(preco,    " R$", 2),
            "P/L":           _fmt(pl,       "",    2),
            "P/VP":          _fmt(pvp,      "",    2),
            "LPA":           _fmt(lpa,      " R$", 2),
            "ROE":           _fmt(roe_pct,  "%",   2),
            "DY":            _fmt(dy_pct,   "%",   2),
            "Margem Líq.":   _fmt(margin_pct,"%",  2),
            "EV/EBITDA":     _fmt(ebitda_m, "",    2),
            "Máx 52s":       _fmt(high52,   " R$", 2),
            "Mín 52s":       _fmt(low52,    " R$", 2),
            "Mkt Cap":       f"R$ {mktcap/1e9:.1f} bi" if isinstance(mktcap, (int, float)) else "N/A",
            "Score":         score,
            # valores numéricos para gráficos
            "_pl":           pl       if isinstance(pl,      float) else None,
            "_pvp":          pvp      if isinstance(pvp,     float) else None,
            "_roe":          roe_pct,
            "_dy":           dy_pct,
        })

    if not financial_data:
        st.error("Nenhum dado foi carregado. Verifique os tickers selecionados.")
        return

    # ── Cards por ativo ──────────────────────────────────────────────
    for co in financial_data:
        with st.container(border=True):
            logo_col, info_col = st.columns([1, 11])
            with logo_col:
                try:
                    st.image(
                        f"https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{co['Ticker']}.png",
                        width=55,
                    )
                except Exception:
                    pass
            with info_col:
                score     = co["Score"]
                score_cls = "score-hi" if score >= 7 else ("score-md" if score >= 4 else "score-lo")
                score_lbl = "Excelente" if score >= 7 else ("Regular" if score >= 4 else "Fraco")

                th, sc = st.columns([9, 1])
                th.subheader(f"{co['Ticker']}  —  {co['Nome']}", divider="red")
                sc.markdown(
                    f'<div style="text-align:center">'
                    f'<span class="{score_cls}">{score}/10</span>'
                    f'<br><small style="color:#64748b">{score_lbl}</small></div>',
                    unsafe_allow_html=True,
                )

            # linha 1: preço e avaliação
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Preço atual",  co["Preço"])
            c2.metric("P/L",          co["P/L"])
            c3.metric("P/VP",         co["P/VP"])
            c4.metric("LPA",          co["LPA"])
            c5.metric("Mkt Cap",      co["Mkt Cap"])

            # linha 2: rentabilidade e outros
            c6, c7, c8, c9, c10 = st.columns(5)
            c6.metric("ROE",          co["ROE"])
            c7.metric("DY",           co["DY"])
            c8.metric("Margem Líq.",  co["Margem Líq."])
            c9.metric("EV/EBITDA",    co["EV/EBITDA"])
            c10.metric("Máx / Mín 52s", f"{co['Máx 52s']} / {co['Mín 52s']}")

    style_metric_cards(background_color="rgba(255,255,255,0)")

    # ── Comparativo visual (só com 2+ ativos) ───────────────────────
    if len(financial_data) > 1:
        st.markdown('<div class="sec-label">Comparativo entre ativos</div>', unsafe_allow_html=True)

        compare_df = pd.DataFrame([{
            "Ticker": co["Ticker"],
            "P/L":    co["_pl"]  or 0,
            "P/VP":   co["_pvp"] or 0,
            "ROE (%)": co["_roe"] or 0,
            "DY (%)":  co["_dy"]  or 0,
        } for co in financial_data])

        tab1, tab2 = st.tabs(["Valuation (P/L e P/VP)", "Rentabilidade (ROE e DY)"])
        with tab1:
            fig = px.bar(
                compare_df.melt(id_vars="Ticker", value_vars=["P/L", "P/VP"]),
                x="Ticker", y="value", color="variable", barmode="group",
                template="plotly_dark", height=360,
                color_discrete_map={"P/L": "#38bdf8", "P/VP": "#a78bfa"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = px.bar(
                compare_df.melt(id_vars="Ticker", value_vars=["ROE (%)", "DY (%)"]),
                x="Ticker", y="value", color="variable", barmode="group",
                template="plotly_dark", height=360,
                color_discrete_map={"ROE (%)": "#22c55e", "DY (%)": "#f59e0b"},
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# ABA: RESULTADOS TRIMESTRAIS
# ─────────────────────────────────────────────

def results_dashboard(tickers):
    st.markdown('<div class="sec-label">📉 Resultados Trimestrais</div>', unsafe_allow_html=True)

    if not tickers:
        st.warning("Selecione pelo menos um ticker.")
        return

    ticker_clean_list = [t.removesuffix(".SA") for t in tickers]
    selected = st.selectbox("Selecione o ativo", ticker_clean_list)
    ticker_yf = selected + ".SA"

    with st.spinner("Carregando resultados..."):
        try:
            qf, qi = fetch_quarterly_financials(ticker_yf)
        except Exception as e:
            st.error(f"Erro ao buscar dados: {e}")
            return

    # Tenta os dois objetos retornados
    fin = None
    for candidate in [qf, qi]:
        if candidate is not None and not (hasattr(candidate, "empty") and candidate.empty):
            fin = candidate
            break

    if fin is None or (hasattr(fin, "empty") and fin.empty):
        st.warning("Dados trimestrais não disponíveis para este ativo no Yahoo Finance.")
        return

    # Transpõe: datas nas linhas, métricas nas colunas
    try:
        df = fin.T.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    except Exception as e:
        st.error(f"Erro ao processar dados: {e}")
        return

    def get_col(candidates):
        for c in candidates:
            matches = [col for col in df.columns if c.lower() in str(col).lower()]
            if matches:
                return matches[0]
        return None

    rev_col    = get_col(["Total Revenue", "Revenue"])
    net_col    = get_col(["Net Income"])
    gross_col  = get_col(["Gross Profit"])

    if rev_col is None and net_col is None:
        st.warning("Não encontrei colunas de Receita ou Lucro neste ativo.")
        with st.expander("Colunas disponíveis"):
            st.write(list(df.columns))
        return

    # Cards resumo
    if rev_col and len(df) > 0:
        last_rev = df[rev_col].dropna().iloc[-1] if len(df[rev_col].dropna()) > 0 else None
        if last_rev:
            last_rev_bi = last_rev / 1e9
            prev_rev    = df[rev_col].dropna().iloc[-2] if len(df[rev_col].dropna()) > 1 else None
            rev_chg     = (last_rev / prev_rev - 1) if prev_rev else None

            c1, c2, c3 = st.columns(3)
            c1.metric("Receita (último trim.)",
                      f"R$ {last_rev_bi:.2f} bi",
                      f"{rev_chg:+.1%}" if rev_chg else None)

            if net_col:
                last_net = df[net_col].dropna().iloc[-1] if len(df[net_col].dropna()) > 0 else None
                if last_net:
                    margin = last_net / last_rev * 100 if last_rev else None
                    c2.metric("Lucro Líq. (último trim.)",
                              f"R$ {last_net/1e9:.2f} bi")
                    c3.metric("Margem Líquida", f"{margin:.1f}%" if margin else "N/A")

            style_metric_cards(background_color="rgba(255,255,255,0)")

    # Gráficos
    charts_available = []
    if rev_col:   charts_available.append(("Receita Total", rev_col,   "#38bdf8"))
    if net_col:   charts_available.append(("Lucro Líquido",  net_col,  "#22c55e"))
    if gross_col: charts_available.append(("Lucro Bruto",    gross_col,"#a78bfa"))

    for label, col, color in charts_available:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        series = series / 1e9
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[str(d.date()) for d in series.index],
            y=series.values,
            marker_color=color,
            name=label,
        ))
        fig.update_layout(
            template="plotly_dark",
            title=f"{selected} — {label} (R$ bilhões)",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# ABA: CORRELAÇÃO
# ─────────────────────────────────────────────

def correlation_dashboard(prices):
    st.markdown('<div class="sec-label">🔗 Correlação Entre Ações</div>', unsafe_allow_html=True)
    corr = prices.drop(columns="IBOV").corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=.5,
                ax=ax, cbar_kws={"shrink": .8}, annot_kws={"color": "black"})
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    plt.xticks(rotation=45, ha="center", color="white")
    plt.yticks(rotation=0, color="white")
    ax.set_title("Heatmap de Correlação", fontsize=14, pad=20, color="white")
    plt.tight_layout()
    st.pyplot(fig)


# ─────────────────────────────────────────────
# ABA: DIVIDENDOS
# ─────────────────────────────────────────────

def dividends_dashboard(tickers):
    st.markdown('<div class="sec-label">💰 Histórico de Dividendos</div>', unsafe_allow_html=True)

    if not tickers:
        st.warning("Selecione pelo menos um ticker.")
        return

    min_year = st.slider("Mostrar a partir do ano", 2000, datetime.now().year, 2015)
    annual = {}

    for t in tickers:
        t_clean = t.removesuffix(".SA")
        try:
            divs = fetch_dividends(t)
            if divs is None or len(divs) == 0:
                annual[t_clean] = pd.Series(dtype=float)
                continue
            divs = divs.copy()
            divs.index = pd.to_datetime(divs.index)
            ann = divs.resample("Y").sum()
            ann.index = ann.index.year
            annual[t_clean] = ann[ann.index >= min_year]
        except Exception as e:
            st.error(f"Erro em {t_clean}: {e}")
            annual[t_clean] = pd.Series(dtype=float)

    if not any(len(s) > 0 for s in annual.values()):
        st.warning("Nenhum histórico de dividendos encontrado.")
        return

    df = pd.DataFrame(annual).fillna(0)
    df.index.name = "Ano"
    st.dataframe(df, use_container_width=True)

    df_plot = df.reset_index().melt(id_vars="Ano", var_name="Ticker", value_name="Dividendos")
    fig = px.bar(df_plot, x="Ano", y="Dividendos", color="Ticker", barmode="group",
                 title="Dividendos por ação (R$ — soma anual)", template="plotly_dark", height=550)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# ABA: AGENDA
# ─────────────────────────────────────────────

def agenda_dashboard(tickers):
    st.markdown('<div class="sec-label">🗓️ Agenda do Mercado</div>', unsafe_allow_html=True)

    tab_copom, tab_earnings, tab_div = st.tabs(["📅 Reuniões COPOM", "📊 Próximos Resultados", "💵 Próximos Dividendos"])

    with tab_copom:
        st.caption("Calendário oficial das reuniões do Comitê de Política Monetária (Banco Central do Brasil).")
        today = datetime.now().date()
        for d1_str, d2_str in COPOM_DATES:
            d1 = datetime.strptime(d1_str, "%Y-%m-%d").date()
            d2 = datetime.strptime(d2_str, "%Y-%m-%d").date()
            is_next = (d1 >= today)
            status  = "🔜 Próxima" if is_next and all(
                datetime.strptime(x[0], "%Y-%m-%d").date() >= today or
                datetime.strptime(x[0], "%Y-%m-%d").date() > d1
                for x in COPOM_DATES if datetime.strptime(x[0], "%Y-%m-%d").date() >= today
                if datetime.strptime(x[0], "%Y-%m-%d").date() == d1
            ) else ("📅 Agendada" if is_next else "✅ Realizada")

            cls = "copom-next" if is_next else "copom-past"
            st.markdown(
                f'<div class="{cls}"><strong>{d1.strftime("%d/%m/%Y")} – {d2.strftime("%d/%m/%Y")}</strong>'
                f'  &nbsp; <small>{status}</small></div>',
                unsafe_allow_html=True
            )

    with tab_earnings:
        if not tickers:
            st.warning("Selecione ativos na barra lateral para ver datas de resultado.")
        else:
            for t in tickers:
                tk = t.removesuffix(".SA")
                try:
                    info = yf.Ticker(t)
                    cal  = info.calendar
                    if cal is not None and "Earnings Date" in cal:
                        dates = cal["Earnings Date"]
                        dates_str = ", ".join(str(d) for d in dates) if hasattr(dates, "__iter__") else str(dates)
                        st.markdown(f"**{tk}** — próximo resultado: `{dates_str}`")
                    else:
                        st.markdown(f"**{tk}** — data de resultado não disponível")
                except Exception:
                    st.markdown(f"**{tk}** — não foi possível obter data de resultado")

    with tab_div:
        if not tickers:
            st.warning("Selecione ativos na barra lateral para ver dividendos futuros.")
        else:
            for t in tickers:
                tk = t.removesuffix(".SA")
                try:
                    info = yf.Ticker(t)
                    cal  = info.calendar
                    if cal is not None and "Ex-Dividend Date" in cal:
                        ex_date = cal["Ex-Dividend Date"]
                        st.markdown(f"**{tk}** — ex-dividendo: `{ex_date}`")
                    else:
                        st.markdown(f"**{tk}** — data ex-dividendo não disponível")
                except Exception:
                    st.markdown(f"**{tk}** — não foi possível obter data de dividendo")


# ─────────────────────────────────────────────
# ABA: WATCHLIST
# ─────────────────────────────────────────────

def watchlist_dashboard():
    st.markdown('<div class="sec-label">⭐ Watchlist</div>', unsafe_allow_html=True)

    wl = load_watchlist()

    col_add, col_btn = st.columns([4, 1])
    new_ticker = col_add.text_input("Adicionar ativo (ex: VALE3)", placeholder="VALE3").upper().strip()
    with col_btn:
        st.write("")
        st.write("")
        if st.button("➕ Adicionar") and new_ticker:
            if new_ticker not in wl:
                wl.append(new_ticker)
                save_watchlist(wl)
                st.rerun()

    if not wl:
        st.info("Sua watchlist está vazia. Adicione ativos acima.")
        return

    tickers_yf = [t + ".SA" for t in wl]
    with st.spinner("Atualizando cotações..."):
        try:
            data = yf.download(tickers_yf, period="2d", progress=False, auto_adjust=True)
            close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
        except Exception:
            close = pd.DataFrame()

    st.markdown("---")
    cols = st.columns(min(4, len(wl)))
    to_remove = []

    for i, ticker in enumerate(wl):
        with cols[i % len(cols)]:
            with st.container(border=True):
                try:
                    ticker_yf = ticker + ".SA"
                    if not close.empty:
                        if len(tickers_yf) == 1:
                            series = close
                        else:
                            series = close[ticker_yf] if ticker_yf in close.columns else pd.Series()
                        if len(series.dropna()) >= 2:
                            last = float(series.dropna().iloc[-1])
                            prev = float(series.dropna().iloc[-2])
                            chg  = (last / prev - 1)
                            st.metric(
                                ticker,
                                f"R$ {last:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                                f"{chg:+.2%}"
                            )
                        else:
                            st.metric(ticker, "—")
                    else:
                        st.metric(ticker, "—")
                except Exception:
                    st.metric(ticker, "erro")

                if st.button("🗑️", key=f"rm_{ticker}", help="Remover"):
                    to_remove.append(ticker)

    if to_remove:
        for t in to_remove:
            wl.remove(t)
        save_watchlist(wl)
        st.rerun()

    style_metric_cards(background_color="rgba(255,255,255,0)")


# ─────────────────────────────────────────────
# ABA: RRG
# ─────────────────────────────────────────────

def rrg_graph(tickers, prices):
    st.markdown('<div class="sec-label">🔄 Relative Rotation Graph (RRG)</div>', unsafe_allow_html=True)

    try:
        setor_data = pd.read_csv("tickers/tickers_setor.csv", encoding="latin1")
        setores    = setor_data["Setor"].unique()

        setor_sel = st.selectbox("Filtrar por Setor", ["Nenhum"] + list(setores))

        tickers_setor = []
        if setor_sel != "Nenhum":
            tickers_setor = [t + ".SA" for t in
                             setor_data[setor_data["Setor"] == setor_sel]["Ticker"].tolist()]

        tickers_filtrados = list(set((tickers or []) + tickers_setor))
        if not tickers_filtrados:
            st.warning("Selecione pelo menos um ativo manualmente ou escolha um setor.")
            return

        end_date   = datetime.now()
        start_date = end_date - pd.DateOffset(years=2)

        with st.spinner("Carregando dados RRG..."):
            pf = download_rrg_prices(tuple(sorted(tickers_filtrados)), start_date, end_date)

        if pf.empty:
            st.error("Não foi possível obter dados.")
            return

        wr = pf.resample("W").last().pct_change().dropna()
        if wr.empty or len(wr) < 10:
            st.error("Dados insuficientes para o RRG.")
            return

        bm  = wr["IBOV"]
        rs  = wr.div(bm, axis=0) - 1
        mom = rs - rs.shift(12)

        rs_n  = (rs  - rs.mean())  / rs.std()
        mom_n = (mom - mom.mean()) / mom.std()

        data = pd.DataFrame({
            "Ticker":           rs_n.columns,
            "Relative Strength": rs_n.iloc[-1].values,
            "Momentum":          mom_n.iloc[-1].values,
        })
        data["Quadrante"] = np.where(
            (data["Relative Strength"] > 0) & (data["Momentum"] > 0), "Líderes",
            np.where(
                (data["Relative Strength"] < 0) & (data["Momentum"] > 0), "Melhorando",
                np.where(
                    (data["Relative Strength"] > 0) & (data["Momentum"] < 0), "Enfraquecendo",
                    "Defasados"
                )
            )
        )

        fig = px.scatter(data, x="Relative Strength", y="Momentum",
                         text="Ticker", color="Quadrante",
                         color_discrete_map={"Líderes":"#22c55e","Melhorando":"#38bdf8",
                                             "Enfraquecendo":"#f97316","Defasados":"#ef4444"},
                         template="plotly_dark",
                         title=f"RRG — Setor: {setor_sel}")
        fig.add_shape(type="line", x0=0, y0=-2.5, x1=0, y1=2.5, line=dict(color="white", dash="dot", width=.8))
        fig.add_shape(type="line", x0=-2.5, y0=0, x1=2.5, y1=0, line=dict(color="white", dash="dot", width=.8))
        for label, x, y, color in [("Líderes", 1.2, 1.5, "#22c55e"), ("Melhorando", -1.5, 1.5, "#38bdf8"),
                                    ("Enfraquecendo", 1.2, -1.5, "#f97316"), ("Defasados", -1.5, -1.5, "#ef4444")]:
            fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(color=color, size=13))
        fig.update_traces(marker=dict(size=15, line=dict(width=1.5, color="DarkSlateGrey")), textposition="top center")
        fig.update_layout(xaxis_range=[-2.5, 2.5], yaxis_range=[-2.5, 2.5], showlegend=False, height=620)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro no RRG: {e}")


# ─────────────────────────────────────────────
# ABA: MAPA IBOVESPA
# ─────────────────────────────────────────────

def ibovespa_map():
    st.markdown('<div class="sec-label">🗺️ Mapa do Ibovespa</div>', unsafe_allow_html=True)

    try:
        tickers_tuple = tuple(t + ".SA" for t in IBOV_COMPOSITION.keys())
        with st.spinner("Obtendo dados em tempo real..."):
            data = download_ibov_map_data(tickers_tuple)

        if data.empty:
            st.error("Dados não disponíveis no momento.")
            return

        plot_data = []
        for ticker, info in IBOV_COMPOSITION.items():
            tk = ticker + ".SA"
            try:
                if tk in data:
                    cd = data[tk]["Close"] if isinstance(data, dict) else data[tk].Close
                    if len(cd) >= 2 and not pd.isna(cd.iloc[-1]):
                        cur  = cd.iloc[-1]
                        prev = cd.iloc[-2]
                        var  = (cur / prev - 1) * 100
                        color = "green" if var >= 0 else "red"
                        plot_data.append({
                            "Ticker":   ticker,
                            "Setor":    info["setor"],
                            "Peso":     info["peso"],
                            "Variação": var,
                            "Texto": (
                                f"<b>{ticker}</b><br>"
                                f"R$ {cur:.2f}<br>"
                                f"<span style='color:{color}'>{var:+.2f}%</span><br>"
                                f"{info['peso']:.2f}%"
                            )
                        })
            except Exception:
                continue

        if not plot_data:
            st.error("Não foi possível obter cotações válidas.")
            return

        df = pd.DataFrame(plot_data)
        fig = px.treemap(df, path=["Setor", "Ticker"], values="Peso", color="Variação",
                         color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                         hover_name="Texto", hover_data={"Texto": False}, height=700)
        fig.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]}",
            textfont=dict(family="Arial Black", color="black", size=13),
            textposition="middle center",
            marker=dict(line=dict(width=1, color="DarkSlateGrey"))
        )
        fig.update_layout(margin=dict(t=30, l=10, r=10, b=10),
                          coloraxis_colorbar=dict(title="Var %", thickness=14))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao gerar o mapa: {e}")


# ─────────────────────────────────────────────
# ABA: FLUXO ESTRANGEIRO
# ─────────────────────────────────────────────

def extract_table_from_report(text, table_title_contains, sep=";"):
    lines = [ln.strip("﻿") for ln in text.splitlines()]
    start_idx = next((i for i, ln in enumerate(lines)
                      if table_title_contains.lower() in ln.lower()), None)
    if start_idx is None:
        return pd.DataFrame()

    header_idx = None
    for j in range(start_idx + 1, min(start_idx + 30, len(lines))):
        if sep in lines[j]:
            parts = [p.strip() for p in lines[j].split(sep)]
            if len(parts) >= 2 and any(k in p.lower() for p in parts for k in ["mês", "mes", "month"]):
                header_idx = j
                break
    if header_idx is None:
        return pd.DataFrame()

    header = [h.strip() for h in lines[header_idx].split(sep)]
    data = []
    for k in range(header_idx + 1, len(lines)):
        ln = lines[k].strip()
        if not ln:
            break
        parts = [p.strip() for p in ln.split(sep)]
        if len(parts) < max(2, len(header) // 2):
            break
        parts = (parts + [""] * len(header))[:len(header)]
        data.append(parts)
    return pd.DataFrame(data, columns=header)


def parse_month_year(value):
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.count("/") >= 2:
        parts = s.split("/")
        m = re.search(r"\d{4}", parts[-1])
        if not m:
            return None
        year = int(m.group())
        month_map = {
            "jan":1,"fev":2,"feb":2,"mar":3,"abr":4,"apr":4,"mai":5,"may":5,
            "jun":6,"jul":7,"ago":8,"aug":8,"set":9,"sep":9,"out":10,"oct":10,
            "nov":11,"dez":12,"dec":12,
        }
        key = parts[0].strip().lower().replace(".", "")
        month = month_map.get(key)
        if not month:
            return None
        return pd.Timestamp(year=year, month=month, day=1)
    for fmt in ("%m/%Y", "%Y-%m"):
        dt = pd.to_datetime(s, format=fmt, errors="coerce")
        if not pd.isna(dt):
            return dt.replace(day=1)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return None if pd.isna(dt) else dt.replace(day=1)


def to_float_br(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return np.nan
    try:
        return float(s.replace(".", "").replace(",", "."))
    except Exception:
        return np.nan


def foreign_flow_dashboard():
    st.markdown('<div class="sec-label">🌍 Fluxo Estrangeiro (B3)</div>', unsafe_allow_html=True)

    with st.spinner("Baixando dados da B3..."):
        text = download_b3_dados_mercado_text()

    candidates = [
        "Estrangeiros Mensal", "Investidores Estrangeiros Mensal",
        "Monthly Financial Movement of Foreign Investors",
        "Movimentação dos Investidores Estrangeiros",
    ]
    df = pd.DataFrame()
    for key in candidates:
        df = extract_table_from_report(text, key)
        if not df.empty:
            break

    if df.empty:
        st.error("Tabela de fluxo estrangeiro não encontrada no relatório da B3.")
        return

    month_col = next((c for c in df.columns if any(k in c.lower() for k in ["mês","mes","month","período"])),
                     df.columns[0])
    buy_col  = next((c for c in df.columns if "compra" in c.lower() or "buy"  in c.lower()), None)
    sell_col = next((c for c in df.columns if "venda"  in c.lower() or "sell" in c.lower()), None)
    net_col  = next((c for c in df.columns if "saldo"  in c.lower() or "net"  in c.lower() or "balance" in c.lower()), None)

    out = pd.DataFrame()
    out["Mes"] = df[month_col].apply(parse_month_year)
    if buy_col:  out["Compra"] = df[buy_col].apply(to_float_br)
    if sell_col: out["Venda"]  = df[sell_col].apply(to_float_br)
    if net_col:  out["Saldo"]  = df[net_col].apply(to_float_br)

    out["Mes"] = pd.to_datetime(out["Mes"], errors="coerce")
    out = out.dropna(subset=["Mes"]).sort_values("Mes")

    if "Saldo" not in out.columns and "Compra" in out.columns and "Venda" in out.columns:
        out["Saldo"] = out["Compra"] - out["Venda"]

    if "Saldo" not in out.columns or out.empty:
        st.error("Não foi possível identificar as colunas de saldo no relatório.")
        return

    min_date, max_date = out["Mes"].min(), out["Mes"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        st.error("Datas inválidas no relatório.")
        return

    default_start = max(max_date - pd.DateOffset(years=5), min_date)
    c1, c2 = st.columns(2)
    start = c1.date_input("De",  value=default_start.date())
    end   = c2.date_input("Até", value=max_date.date())

    outf = out.loc[(out["Mes"].dt.date >= start) & (out["Mes"].dt.date <= end)].copy()
    if outf.empty:
        st.warning("Sem dados no intervalo selecionado.")
        return

    last = outf.iloc[-1]
    ytd  = outf[outf["Mes"].dt.year == outf["Mes"].dt.year.max()]["Saldo"].sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Último mês (saldo)",  f"R$ {last['Saldo']:,.0f} mi")
    k2.metric("Acumulado no ano",    f"R$ {ytd:,.0f} mi")
    k3.metric("Meses exibidos",      str(len(outf)))
    style_metric_cards(background_color="rgba(255,255,255,0)")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=outf["Mes"], y=outf["Saldo"], name="Saldo (R$ mi)",
                         marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in outf["Saldo"]]))
    if "Compra" in outf.columns:
        fig.add_trace(go.Scatter(x=outf["Mes"], y=outf["Compra"], name="Compra", mode="lines",
                                 line=dict(color="#38bdf8")))
    if "Venda" in outf.columns:
        fig.add_trace(go.Scatter(x=outf["Mes"], y=outf["Venda"], name="Venda", mode="lines",
                                 line=dict(color="#f97316")))
    fig.update_layout(template="plotly_dark", height=600,
                      title="Fluxo Estrangeiro — mês a mês (B3, R$ milhões)",
                      margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabela de dados"):
        st.dataframe(outf, use_container_width=True)


# ─────────────────────────────────────────────
# ABA: NOTÍCIAS
# ─────────────────────────────────────────────

def news_terminal():
    st.markdown('<div class="sec-label">📰 Notícias do Mercado</div>', unsafe_allow_html=True)

    sources = {
        "Infomoney":       "https://www.infomoney.com.br/feed/",
        "Valor Econômico": "https://valor.globo.com/financas/rss",
        "Investing.com BR":"https://br.investing.com/rss/news.rss",
    }
    c1, c2 = st.columns([3, 1])
    selected_source = c1.selectbox("Fonte", list(sources.keys()))
    num_news = c2.number_input("Qtd.", min_value=5, max_value=30, value=10)

    try:
        with st.spinner("Carregando notícias..."):
            feed = feedparser.parse(sources[selected_source])

        if not feed.entries:
            st.warning("Não foi possível carregar notícias. Tente outra fonte.")
            return

        for entry in feed.entries[:num_news]:
            with st.container(border=True):
                title     = entry.get("title", "Sem título")
                link      = entry.get("link", "")
                published = entry.get("published", "")
                summary   = entry.get("summary", "")

                col1, col2 = st.columns([8, 1])
                with col1:
                    st.markdown(f"**{title}**")
                    if summary:
                        clean = BeautifulSoup(summary, "html.parser").get_text()
                        st.caption(clean[:220] + "..." if len(clean) > 220 else clean)
                    if published:
                        st.caption(f"🕐 {published}")
                with col2:
                    if link:
                        st.link_button("Ver →", link)

    except Exception as e:
        st.error(f"Erro ao carregar notícias: {e}")


# ─────────────────────────────────────────────
# APP PRINCIPAL
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Renova Invest",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

TABS_WITH_TICKERS = ["Dashboard", "Correlação", "Múltiplos", "Resultados", "Dividendos", "Agenda", "RRG"]

with st.sidebar:
    selected_tab = st.radio(
        "Navegação",
        ["📊 Dashboard", "📈 Análise Técnica", "🔍 Screener",
         "📋 Múltiplos", "📉 Resultados", "🔗 Correlação",
         "💰 Dividendos", "🌍 Fluxo Estrangeiro", "🗓️ Agenda",
         "⭐ Watchlist", "🔄 RRG", "🗺️ Mapa Ibovespa", "📰 Notícias"],
        label_visibility="collapsed",
    )
    tab_key = selected_tab.split(" ", 1)[1]  # remove emoji prefix
    st.markdown("---")
    if tab_key in TABS_WITH_TICKERS:
        tickers, prices = build_sidebar()
    else:
        tickers, prices = None, None

# Header
st.markdown(
    '<div class="main-title">Renova Invest</div>'
    '<div class="main-subtitle">Mercado de Capitais · B3</div>',
    unsafe_allow_html=True
)

# Roteamento
NEEDS_TICKERS = {"Dashboard", "Correlação", "Múltiplos", "Resultados", "Dividendos", "Agenda", "RRG"}

if tab_key in NEEDS_TICKERS and (not tickers or prices is None):
    st.warning("Selecione pelo menos um ticker na barra lateral.")
elif tab_key == "Dashboard":
    main_dashboard(tickers, prices)
elif tab_key == "Análise Técnica":
    technical_analysis_dashboard()
elif tab_key == "Screener":
    screener_dashboard()
elif tab_key == "Múltiplos":
    multiples_dashboard(tickers)
elif tab_key == "Resultados":
    results_dashboard(tickers)
elif tab_key == "Correlação":
    correlation_dashboard(prices)
elif tab_key == "Dividendos":
    dividends_dashboard(tickers)
elif tab_key == "Fluxo Estrangeiro":
    foreign_flow_dashboard()
elif tab_key == "Agenda":
    agenda_dashboard(tickers)
elif tab_key == "Watchlist":
    watchlist_dashboard()
elif tab_key == "RRG":
    rrg_graph(tickers, prices)
elif tab_key == "Mapa Ibovespa":
    ibovespa_map()
elif tab_key == "Notícias":
    news_terminal()
