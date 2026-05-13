import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid
import seaborn as sns
import matplotlib.pyplot as plt
import yahooquery as yq
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import requests
import re
import feedparser


# ─────────────────────────────────────────────
# FUNÇÕES DE DOWNLOAD COM CACHE
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def download_prices(tickers: tuple, start_date, end_date) -> pd.DataFrame:
    """Baixa preços ajustados + IBOV para os tickers informados."""
    prices = yf.download(list(tickers), start=start_date, end=end_date, progress=False)
    if prices.empty:
        return pd.DataFrame()

    prices = prices["Adj Close"] if "Adj Close" in prices else prices["Close"]

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
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    for col in required + (["Volume"] if "Volume" in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=required).sort_index()


@st.cache_data(ttl=3600)
def fetch_dividends(ticker: str) -> pd.Series:
    return yf.Ticker(ticker).dividends


@st.cache_data(ttl=300)
def download_rrg_prices(tickers_tuple: tuple, start_date, end_date) -> pd.DataFrame:
    tickers = list(tickers_tuple)
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if data.empty:
        return pd.DataFrame()

    prices = data["Adj Close"] if "Adj Close" in data else data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        prices.columns = [tickers[0].removesuffix(".SA")]
    else:
        prices.columns = [c.removesuffix(".SA") for c in prices.columns]

    ibov = yf.download("^BVSP", start=start_date, end=end_date, progress=False)
    prices["IBOV"] = ibov["Adj Close"] if "Adj Close" in ibov else ibov["Close"]
    return prices


@st.cache_data(ttl=300)
def download_ibov_map_data(tickers_tuple: tuple) -> pd.DataFrame:
    return yf.download(list(tickers_tuple), period="2d", group_by="ticker", progress=False)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def build_sidebar():
    st.image("images/avatar-renova-instagram.png")
    ticker_list = pd.read_csv("tickers/tickers_ibra.csv", index_col=0)

    tickers = st.multiselect(label="Selecione as Empresas", options=ticker_list, placeholder='Códigos')
    tickers = [t + ".SA" for t in tickers]
    start_date = st.date_input("De", value=datetime(2023, 1, 2), format="YYYY-MM-DD")
    end_date = st.date_input("Até", value=datetime.now().date(), format="YYYY-MM-DD")

    if not tickers:
        st.warning("Por favor, selecione pelo menos um ticker.")
        return None, None

    with st.spinner("Carregando dados..."):
        prices = download_prices(tuple(tickers), start_date, end_date)

    if prices.empty:
        st.error("Não foi possível obter dados. Verifique os códigos ou tente novamente.")
        return None, None

    return tickers, prices


# ─────────────────────────────────────────────
# CÁLCULOS
# ─────────────────────────────────────────────

def calculate_beta(returns, market_returns):
    covariance = np.cov(returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance


def calculate_correlation(prices):
    return prices.corr()


# ─────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────

def main_dashboard(tickers, prices):
    weights = np.ones(len(tickers)) / len(tickers)
    prices['portfolio'] = prices.drop("IBOV", axis=1) @ weights
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
            colA.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{t}.png', width=85, use_container_width=True)

        colB.metric(label="Retorno", value=f"{rets[t]:.0%}")
        colC.metric(label="Volatilidade", value=f"{vols[t]:.0%}")
        colD.metric(label="Beta", value=f"{betas[t]:.2f}" if t in betas else "N/A")

        style_metric_cards(background_color='rgba(255,255,255,0)')

    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.subheader("Desempenho Relativo")
        st.line_chart(norm_prices, height=600)

    with col2:
        st.subheader("Risco-Retorno")
        fig = px.scatter(
            x=vols,
            y=rets,
            text=vols.index,
            color=rets / vols,
            color_continuous_scale="Blues",
            labels={"x": "Volatilidade", "y": "Retorno"},
            template="plotly_dark",
            title="Relação Risco-Retorno"
        )
        fig.update_traces(
            textfont_color='black',
            marker=dict(size=42, line=dict(width=1, color="DarkSlateGrey")),
            textfont_size=12,
        )
        fig.layout.height = 600
        fig.layout.coloraxis.colorbar.title = 'Sharpe'
        st.plotly_chart(fig, use_container_width=True)


def technical_analysis_dashboard():
    st.subheader("📈 Análise Técnica (Candles)")

    ticker_list = pd.read_csv("tickers/tickers_ibra.csv", index_col=0)
    ticker = st.selectbox("Ticker (1 por vez)", options=ticker_list)
    ticker_yf = f"{ticker}.SA"

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        interval_label = st.selectbox("Timeframe", ["Diário", "Semanal", "Mensal"], index=0)
    with col2:
        period = st.selectbox("Período", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
    with col3:
        show_volume = st.toggle("Mostrar volume", value=True)

    interval = {"Diário": "1d", "Semanal": "1wk", "Mensal": "1mo"}[interval_label]

    with st.spinner("Carregando candles..."):
        df = fetch_ohlc(ticker_yf, period=period, interval=interval)

    if df is None or df.empty:
        st.error("Não foi possível carregar dados para este ativo/período.")
        return

    try:
        last_close = float(df["Close"].iloc[-1])
        first_close = float(df["Close"].iloc[0])
        period_change = (last_close / first_close) - 1
        last_change = (last_close / float(df["Close"].iloc[-2]) - 1) if len(df) >= 2 else np.nan

        cA, cB, cC = st.columns(3)
        cA.metric("Preço atual (Close)", f"R$ {last_close:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        cB.metric(f"Variação do período ({period})", f"{period_change:+.2%}")
        cC.metric(
            f"Variação do último candle ({interval_label})",
            f"{last_change:+.2%}" if pd.notna(last_change) else "—"
        )
    except Exception as e:
        st.warning(f"Não foi possível calcular preço/variações: {e}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index.to_pydatetime(),
        open=df["Open"].values,
        high=df["High"].values,
        low=df["Low"].values,
        close=df["Close"].values,
        name=ticker
    ))

    if show_volume and "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.25, yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"))

    fig.update_layout(
        template="plotly_dark",
        height=700,
        title=f"{ticker} • {interval_label} • {period}",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    try:
        ret = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0])) - 1
        amp = (float(df["High"].max()) / float(df["Low"].min())) - 1
        c1, c2 = st.columns(2)
        c1.metric("Retorno no período", f"{ret:.2%}")
        c2.metric("Amplitude (High / Low)", f"{amp:.2%}")
    except Exception as e:
        st.warning(f"Não foi possível calcular métricas do período: {e}")


def correlation_dashboard(prices):
    correlation_matrix = calculate_correlation(prices.drop(columns="IBOV"))
    st.subheader("Correlação Entre Ações Selecionadas")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"color": "black"},
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xticks(rotation=45, ha="center", color="white")
    plt.yticks(rotation=0, color="white")
    ax.set_title("Heatmap de Correlação", loc="center", fontsize=14, pad=20, color="white")
    plt.tight_layout()
    st.pyplot(fig)


def multiples_dashboard(tickers):
    """Exibe múltiplos financeiros de forma robusta e integrada ao layout."""
    st.subheader("Múltiplos Financeiros")

    if not tickers:
        st.warning("Por favor, insira pelo menos um ticker.")
        return

    try:
        ticker_data = yq.Ticker(tickers)
        summary_data = ticker_data.summary_detail
        key_stats_data = ticker_data.key_stats

        financial_data = []
        for ticker in tickers:
            ticker_clean = ticker.removesuffix(".SA")
            summary = summary_data.get(ticker, {})
            key_stats = key_stats_data.get(ticker, {})

            try:
                net_income = key_stats.get("netIncomeToCommon")
                book_value_per_share = key_stats.get("bookValue")
                shares_outstanding = key_stats.get("sharesOutstanding")
                total_equity = (
                    book_value_per_share * shares_outstanding
                    if book_value_per_share and shares_outstanding
                    else None
                )
                roe = (
                    f"{(net_income / total_equity) * 100:.2f}%"
                    if net_income and total_equity
                    else "Dado não disponível"
                )

                financial_data.append({
                    "Ticker": ticker_clean,
                    "P/L": summary.get("trailingPE", "Indisponível"),
                    "P/VP": key_stats.get("priceToBook", "Indisponível"),
                    "LPA": key_stats.get("trailingEps", "Indisponível"),
                    "Margem Bruta": f"{key_stats.get('profitMargins', 0) * 100:.2f}%" if key_stats.get('profitMargins') is not None else "Indisponível",
                    "Margem EBITDA": f"{key_stats.get('enterpriseToEbitda', 0):.2f}" if key_stats.get('enterpriseToEbitda') is not None else "Indisponível",
                    "ROE": roe,
                    "Dividend Yield": f"{summary.get('dividendYield', 0) * 100:.2f}%" if summary.get('dividendYield') is not None else "Indisponível",
                })
            except Exception as e:
                st.error(f"Erro ao processar dados de {ticker_clean}: {e}")

        for company in financial_data:
            with st.container():
                st.write("---")
                cols = st.columns([2, 2, 2, 2, 2, 2])
                with cols[0]:
                    st.image(
                        f"https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{company['Ticker']}.png",
                        width=120,
                        caption=company['Ticker']
                    )
                with cols[1]:
                    st.metric(label="P/L", value=company["P/L"])
                with cols[2]:
                    st.metric(label="P/VP", value=company["P/VP"])
                with cols[3]:
                    st.metric(label="LPA", value=company["LPA"])
                with cols[4]:
                    st.metric(label="Margem Bruta", value=company["Margem Bruta"])
                with cols[5]:
                    st.metric(label="ROE", value=company["ROE"])

                cols2 = st.columns([2, 2, 2, 2])
                with cols2[0]:
                    st.metric(label="Margem EBITDA", value=company["Margem EBITDA"])
                with cols2[1]:
                    st.metric(label="Dividend Yield", value=company["Dividend Yield"])

        style_metric_cards(background_color='rgba(255,255,255,0)')

    except Exception as e:
        st.error(f"Erro geral ao obter dados financeiros: {e}")


def dividends_dashboard(tickers):
    """Mostra histórico de dividendos (por ação) em barras ano a ano."""
    st.subheader("📊 Histórico de Dividendos (ano a ano)")

    if not tickers:
        st.warning("Selecione pelo menos um ticker.")
        return

    min_year = st.slider("Mostrar a partir do ano", 2000, datetime.now().year, 2015)

    annual = {}
    for t in tickers:
        t_clean = t.replace(".SA", "")
        try:
            divs = fetch_dividends(t)

            if divs is None or len(divs) == 0:
                annual[t_clean] = pd.Series(dtype=float)
                continue

            divs = divs.copy()
            divs.index = pd.to_datetime(divs.index)
            annual_sum = divs.resample("Y").sum()
            annual_sum.index = annual_sum.index.year
            annual[t_clean] = annual_sum[annual_sum.index >= min_year]

        except Exception as e:
            st.error(f"Erro ao obter dividendos de {t_clean}: {e}")
            annual[t_clean] = pd.Series(dtype=float)

    if not any(len(s) > 0 for s in annual.values()):
        st.warning("Não encontrei histórico de dividendos para os ativos selecionados.")
        return

    df = pd.DataFrame(annual).fillna(0)
    df.index.name = "Ano"
    st.dataframe(df, use_container_width=True)

    df_plot = df.reset_index().melt(id_vars="Ano", var_name="Ticker", value_name="Dividendos")
    fig = px.bar(
        df_plot,
        x="Ano",
        y="Dividendos",
        color="Ticker",
        barmode="group",
        title="Dividendos por ação (soma no ano)",
        template="plotly_dark",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def rrg_graph(tickers, prices):
    """Gera o gráfico RRG (Relative Rotation Graph) com filtro por setor e/ou tickers selecionados."""
    st.subheader("Relative Rotation Graph (RRG)")

    try:
        setor_data = pd.read_csv("tickers/tickers_setor.csv", encoding='latin1')
        setores = setor_data["Setor"].unique()

        setor_selecionado = st.selectbox("Selecione o Setor", ["Nenhum"] + list(setores))

        if setor_selecionado == "Todos":
            tickers_setor = []
        else:
            tickers_setor = setor_data[setor_data["Setor"] == setor_selecionado]["Ticker"].tolist()
            tickers_setor = [t + ".SA" for t in tickers_setor]

        tickers_filtrados = list(set(tickers + tickers_setor)) if tickers else tickers_setor

        if not tickers_filtrados:
            st.warning("Selecione pelo menos um ativo manualmente ou escolha um setor.")
            return

        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(years=2)

        with st.spinner("Carregando dados RRG..."):
            prices_filtrados = download_rrg_prices(tuple(sorted(tickers_filtrados)), start_date, end_date)

        if prices_filtrados.empty:
            st.error("Não foi possível obter dados para os tickers selecionados.")
            return

        weekly_prices = prices_filtrados.resample('W').last()
        weekly_returns = weekly_prices.pct_change().dropna()

        if weekly_returns.empty or len(weekly_returns) < 10:
            st.error("Não há dados suficientes para calcular os retornos semanais.")
            return

        benchmark_returns = weekly_returns["IBOV"]
        relative_strength = weekly_returns.div(benchmark_returns, axis=0) - 1

        lookback_period = 12
        momentum = relative_strength - relative_strength.shift(lookback_period)

        relative_strength_norm = (relative_strength - relative_strength.mean()) / relative_strength.std()
        momentum_norm = (momentum - momentum.mean()) / momentum.std()

        if relative_strength_norm.empty or momentum_norm.empty:
            st.error("Não há dados suficientes para gerar o gráfico RRG.")
            return

        data = pd.DataFrame({
            "Ticker": relative_strength_norm.columns,
            "Relative Strength": relative_strength_norm.iloc[-1].values,
            "Momentum": momentum_norm.iloc[-1].values,
        })

        data['Quadrante'] = np.where(
            (data['Relative Strength'] > 0) & (data['Momentum'] > 0), "Líderes",
            np.where(
                (data['Relative Strength'] < 0) & (data['Momentum'] > 0), "Melhorando",
                np.where(
                    (data['Relative Strength'] > 0) & (data['Momentum'] < 0), "Enfraquecendo",
                    "Defasados"
                )
            )
        )

        quadrant_colors = {"Líderes": "green", "Melhorando": "blue", "Enfraquecendo": "orange", "Defasados": "red"}

        fig = px.scatter(
            data,
            x="Relative Strength",
            y="Momentum",
            text="Ticker",
            color="Quadrante",
            color_discrete_map=quadrant_colors,
            labels={"Relative Strength": "Força Relativa", "Momentum": "Momentum"},
            title=f"Relative Rotation Graph (RRG) - Setor: {setor_selecionado}",
            template="plotly_dark",
        )

        fig.add_shape(type="line", x0=0, y0=-2, x1=0, y1=2, line=dict(color="white", dash="dash"))
        fig.add_shape(type="line", x0=-2, y0=0, x1=2, y1=0, line=dict(color="white", dash="dash"))
        fig.add_annotation(x=1, y=1, text="Líderes", showarrow=False, font=dict(color="green", size=14))
        fig.add_annotation(x=-1, y=1, text="Melhorando", showarrow=False, font=dict(color="blue", size=14))
        fig.add_annotation(x=1, y=-1, text="Enfraquecendo", showarrow=False, font=dict(color="orange", size=14))
        fig.add_annotation(x=-1, y=-1, text="Defasados", showarrow=False, font=dict(color="red", size=14))

        fig.update_traces(marker=dict(size=15, line=dict(width=2, color="DarkSlateGrey")), textposition="top center")
        fig.update_layout(xaxis_range=[-2, 2], yaxis_range=[-2, 2], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo CSV ou gerar o gráfico RRG: {e}")


def get_ibovespa_composition():
    return {
        "VALE3": {"peso": 10.5, "setor": "Materiais Básicos"},
        "PETR4": {"peso": 8.2, "setor": "Petróleo e Gás"},
        "ITUB4": {"peso": 7.8, "setor": "Financeiro"},
        "BBDC4": {"peso": 5.3, "setor": "Financeiro"},
        "PETR3": {"peso": 5.1, "setor": "Petróleo e Gás"},
        "B3SA3": {"peso": 4.9, "setor": "Financeiro"},
        "ABEV3": {"peso": 3.7, "setor": "Consumo"},
        "BBAS3": {"peso": 3.5, "setor": "Financeiro"},
        "WEGE3": {"peso": 3.2, "setor": "Industrial"},
        "RENT3": {"peso": 2.9, "setor": "Consumo"},
        "SUZB3": {"peso": 2.8, "setor": "Materiais Básicos"},
        "ELET3": {"peso": 2.7, "setor": "Utilidade Pública"},
        "GGBR4": {"peso": 2.6, "setor": "Materiais Básicos"},
        "JBSS3": {"peso": 2.5, "setor": "Consumo"},
        "RADL3": {"peso": 2.4, "setor": "Consumo"},
        "LREN3": {"peso": 2.3, "setor": "Consumo"},
        "HAPV3": {"peso": 2.2, "setor": "Saúde"},
        "PRIO3": {"peso": 2.1, "setor": "Petróleo e Gás"},
        "RAIL3": {"peso": 2.0, "setor": "Industrial"},
        "CCRO3": {"peso": 1.9, "setor": "Industrial"},
    }


def ibovespa_map():
    """Mapa do Ibovespa com variação diária e formatação aprimorada."""
    st.subheader("🗺️ Mapa do Ibovespa - Composição por Setor")

    try:
        composition = get_ibovespa_composition()
        tickers_tuple = tuple(t + ".SA" for t in composition.keys())

        with st.spinner("Obtendo dados em tempo real..."):
            data = download_ibov_map_data(tickers_tuple)

        if data.empty:
            st.error("Dados não disponíveis no momento. Tente novamente mais tarde.")
            return

        plot_data = []
        for ticker in composition.keys():
            ticker_key = ticker + ".SA"
            try:
                if ticker_key in data:
                    close_data = data[ticker_key]["Close"] if isinstance(data, dict) else data[ticker_key].Close

                    if len(close_data) >= 2 and not pd.isna(close_data.iloc[-1]):
                        current_price = close_data.iloc[-1]
                        previous_price = close_data.iloc[-2]
                        variation = ((current_price - previous_price) / previous_price) * 100
                        color = "green" if variation >= 0 else "red"

                        plot_data.append({
                            "Ticker": ticker,
                            "Setor": composition[ticker]["setor"],
                            "Peso": composition[ticker]["peso"],
                            "Preço": current_price,
                            "Variação": variation,
                            "Texto": (
                                f"<b style='font-size:16px'>{ticker}</b><br>"
                                f"<span style='font-size:14px'>R$ {current_price:.2f}</span><br>"
                                f"<span style='font-size:12px; color:{color}'>{variation:+.2f}%</span><br>"
                                f"<span style='font-size:10px'>{composition[ticker]['peso']:.2f}%</span>"
                            )
                        })
            except Exception:
                continue

        if not plot_data:
            st.error("Não foi possível obter cotações válidas.")
            return

        df = pd.DataFrame(plot_data)
        fig = px.treemap(
            df,
            path=['Setor', 'Ticker'],
            values='Peso',
            color='Variação',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            hover_name='Texto',
            hover_data={'Texto': False},
            width=1000,
            height=700
        )
        fig.update_traces(
            texttemplate='<b>%{label}</b><br>%{customdata[0]}',
            textfont=dict(family="Arial Black", color="black", size=14),
            textposition="middle center",
            marker=dict(line=dict(width=1, color='DarkSlateGrey'))
        )
        fig.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),
            uniformtext=dict(minsize=12, mode='hide'),
            coloraxis_colorbar=dict(title="Variação (%)", tickprefix="%", thickness=15)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao gerar o mapa: {str(e)}")


# ─────────────────────────────────────────────
# FLUXO ESTRANGEIRO
# ─────────────────────────────────────────────

B3_DADOS_MERCADO_URL = "https://sistemaswebb3-listados.b3.com.br/marketDataProxy/MarketDataCall/GetDownloadMarketData/RELATORIO_DADOS_DE_MERCADO.csv"


@st.cache_data(ttl=3600)
def download_b3_dados_mercado_text():
    r = requests.get(B3_DADOS_MERCADO_URL, timeout=30)
    r.raise_for_status()
    return r.text


def extract_table_from_report(text, table_title_contains, sep=";"):
    lines = [ln.strip("﻿") for ln in text.splitlines()]
    start_idx = None

    for i, ln in enumerate(lines):
        if table_title_contains.lower() in ln.lower():
            start_idx = i
            break

    if start_idx is None:
        return pd.DataFrame()

    header_idx = None
    for j in range(start_idx + 1, min(start_idx + 30, len(lines))):
        if sep in lines[j]:
            parts = [p.strip() for p in lines[j].split(sep)]
            if len(parts) >= 2 and any(("mês" in p.lower() or "mes" in p.lower() or "month" in p.lower()) for p in parts):
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
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        if len(parts) > len(header):
            parts = parts[:len(header)]
        data.append(parts)

    return pd.DataFrame(data, columns=header)


def parse_month_year(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    if s.count("/") >= 2:
        parts = s.split("/")
        year_str = parts[-1]
        month_str = parts[0]
        m = re.search(r"\d{4}", year_str)
        if not m:
            return None
        year = int(m.group())
        month_map = {
            "jan": 1, "janeiro": 1,
            "fev": 2, "feb": 2, "fevereiro": 2,
            "mar": 3, "march": 3, "março": 3,
            "abr": 4, "apr": 4, "abril": 4,
            "mai": 5, "may": 5, "maio": 5,
            "jun": 6, "june": 6, "junho": 6,
            "jul": 7, "july": 7, "julho": 7,
            "ago": 8, "aug": 8, "agosto": 8,
            "set": 9, "sep": 9, "setembro": 9,
            "out": 10, "oct": 10, "outubro": 10,
            "nov": 11, "november": 11, "novembro": 11,
            "dez": 12, "dec": 12, "december": 12, "dezembro": 12,
        }
        key = month_str.strip().lower().replace(".", "")
        month = month_map.get(key)
        if not month:
            return None
        return pd.Timestamp(year=year, month=month, day=1)

    for fmt in ("%m/%Y", "%Y-%m"):
        dt = pd.to_datetime(s, format=fmt, errors="coerce")
        if not pd.isna(dt):
            return dt.replace(day=1)

    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    return dt.replace(day=1)


def to_float_br(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def foreign_flow_dashboard():
    st.subheader("🌍 Fluxo Estrangeiro (B3) — mês a mês")

    with st.spinner("Baixando dados da B3..."):
        text = download_b3_dados_mercado_text()

    candidates = [
        "Estrangeiros Mensal",
        "Investidores Estrangeiros Mensal",
        "Monthly Financial Movement of Foreign Investors",
        "Movimentação dos Investidores Estrangeiros"
    ]

    df = pd.DataFrame()
    for key in candidates:
        df = extract_table_from_report(text, table_title_contains=key, sep=";")
        if not df.empty:
            break

    if df.empty:
        st.error("Não encontrei a tabela de 'Movimentação dos Investidores Estrangeiros Mensal' no arquivo da B3.")
        return

    month_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["mês", "mes", "month", "período", "period"])),
        df.columns[0]
    )
    buy_col = next((c for c in df.columns if "compra" in c.lower() or "buy" in c.lower()), None)
    sell_col = next((c for c in df.columns if "venda" in c.lower() or "sell" in c.lower()), None)
    net_col = next((c for c in df.columns if "saldo" in c.lower() or "net" in c.lower() or "balance" in c.lower()), None)

    out = pd.DataFrame()
    out["Mes"] = df[month_col].apply(parse_month_year)
    if buy_col:  out["Compra"] = df[buy_col].apply(to_float_br)
    if sell_col: out["Venda"] = df[sell_col].apply(to_float_br)
    if net_col:  out["Saldo"] = df[net_col].apply(to_float_br)

    out["Mes"] = pd.to_datetime(out["Mes"], errors="coerce")
    out = out.dropna(subset=["Mes"]).sort_values("Mes")

    if "Saldo" not in out.columns and "Compra" in out.columns and "Venda" in out.columns:
        out["Saldo"] = out["Compra"] - out["Venda"]

    if "Saldo" not in out.columns:
        st.error("A tabela foi lida, mas não consegui identificar as colunas de Compra/Venda/Saldo.")
        with st.expander("Debug (prévia da tabela)"):
            st.dataframe(df.head(30), use_container_width=True)
        return

    if out.empty:
        st.error("Não foi possível interpretar as datas do relatório da B3.")
        return

    min_date = out["Mes"].min()
    max_date = out["Mes"].max()

    if pd.isna(min_date) or pd.isna(max_date):
        st.error("Datas inválidas no relatório (min/max).")
        return

    default_start = max(max_date - pd.DateOffset(years=5), min_date)
    start = st.date_input("De", value=default_start.date())
    end = st.date_input("Até", value=max_date.date())

    mask = (out["Mes"].dt.date >= start) & (out["Mes"].dt.date <= end)
    outf = out.loc[mask].copy()

    if outf.empty:
        st.warning("Sem dados no intervalo selecionado.")
        return

    last = outf.iloc[-1]
    ytd = outf[outf["Mes"].dt.year == outf["Mes"].dt.year.max()]["Saldo"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Último mês (saldo)", f"R$ {last['Saldo']:,.0f} mi")
    c2.metric("Acumulado no ano", f"R$ {ytd:,.0f} mi")
    c3.metric("Meses exibidos", f"{len(outf)}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=outf["Mes"], y=outf["Saldo"], name="Saldo (R$ mi)"))
    if "Compra" in outf.columns:
        fig.add_trace(go.Scatter(x=outf["Mes"], y=outf["Compra"], name="Compra", mode="lines"))
    if "Venda" in outf.columns:
        fig.add_trace(go.Scatter(x=outf["Mes"], y=outf["Venda"], name="Venda", mode="lines"))

    fig.update_layout(
        template="plotly_dark",
        height=650,
        title="Fluxo Estrangeiro — mês a mês (B3, R$ milhões)",
        xaxis_title="Mês",
        yaxis_title="R$ milhões",
        barmode="relative",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabela"):
        st.dataframe(outf, use_container_width=True)


# ─────────────────────────────────────────────
# NOTÍCIAS
# ─────────────────────────────────────────────

def news_terminal():
    st.subheader("📰 Notícias do Mercado")

    sources = {
        "Infomoney": "https://www.infomoney.com.br/feed/",
        "Valor Econômico": "https://valor.globo.com/financas/rss",
        "Investing.com BR": "https://br.investing.com/rss/news.rss",
    }

    selected_source = st.selectbox("Fonte de notícias", list(sources.keys()))
    feed_url = sources[selected_source]

    try:
        with st.spinner("Carregando notícias..."):
            feed = feedparser.parse(feed_url)

        if not feed.entries:
            st.warning("Não foi possível carregar notícias desta fonte. Tente outra.")
            return

        num_news = st.slider("Número de notícias", 5, 30, 10)

        for entry in feed.entries[:num_news]:
            with st.container(border=True):
                title = entry.get("title", "Sem título")
                link = entry.get("link", "")
                published = entry.get("published", "")
                summary = entry.get("summary", "")

                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(f"**{title}**")
                    if summary:
                        clean = BeautifulSoup(summary, "html.parser").get_text()
                        st.caption(clean[:200] + "..." if len(clean) > 200 else clean)
                    if published:
                        st.caption(f"🕐 {published}")
                with col2:
                    if link:
                        st.link_button("Ver →", link)

    except Exception as e:
        st.error(f"Erro ao carregar notícias: {e}")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

st.set_page_config(layout="wide")

with st.sidebar:
    selected_tab = st.radio(
        "Escolha a visualização",
        ["Dashboard", "Correlação", "Múltiplos", "Fluxo estrangeiro", "Dividendos",
         "Análise Técnica", "Notícias", "RRG", "Mapa Ibovespa"]
    )

    if selected_tab in ["Dashboard", "Correlação", "Múltiplos", "Dividendos", "RRG"]:
        tickers, prices = build_sidebar()
    else:
        tickers, prices = None, None

st.title('Renova Invest - Mercado de Capitais')

if selected_tab == "Mapa Ibovespa":
    ibovespa_map()
elif selected_tab == "Dashboard":
    if tickers and prices is not None:
        main_dashboard(tickers, prices)
    else:
        st.warning("Por favor, selecione pelo menos um ticker na barra lateral.")
elif selected_tab == "Correlação":
    if tickers and prices is not None:
        correlation_dashboard(prices)
    else:
        st.warning("Por favor, selecione pelo menos um ticker na barra lateral.")
elif selected_tab == "Múltiplos":
    if tickers and prices is not None:
        multiples_dashboard(tickers)
    else:
        st.warning("Por favor, selecione pelo menos um ticker na barra lateral.")
elif selected_tab == "Dividendos":
    if tickers and prices is not None:
        dividends_dashboard(tickers)
    else:
        st.warning("Por favor, selecione pelo menos um ticker na barra lateral.")
elif selected_tab == "Análise Técnica":
    technical_analysis_dashboard()
elif selected_tab == "Notícias":
    news_terminal()
elif selected_tab == "Fluxo estrangeiro":
    foreign_flow_dashboard()
elif selected_tab == "RRG":
    if tickers and prices is not None:
        rrg_graph(tickers, prices)
    else:
        st.warning("Por favor, selecione pelo menos um ticker na barra lateral.")
