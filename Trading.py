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


def build_sidebar():
    # Carrega a imagem do repositório (caminho relativo)
    st.image("images/avatar-renova-instagram.png")  # Certifique-se de que a imagem está na pasta "images" do repositório

    # Carrega a lista de tickers do repositório (caminho relativo)
    ticker_list = pd.read_csv("tickers/tickers_ibra.csv", index_col=0)  # Certifique-se de que o CSV está na pasta "tickers"

    # Seleciona os tickers

    tickers = st.multiselect(label="Selecione as Empresas", options=ticker_list, placeholder='Códigos')
    tickers = [t + ".SA" for t in tickers]
    start_date = st.date_input("De", value=datetime(2023, 1, 2), format="YYYY-MM-DD")
    end_date = st.date_input("Até", value=datetime.now().date(), format="YYYY-MM-DD")

    if not tickers:
        st.warning("Por favor, selecione pelo menos um ticker.")
        return None, None

    # Baixa os preços dos tickers selecionados
    prices = yf.download(tickers, start=start_date, end=end_date)

    if prices.empty:
        st.error("Não foi possível obter dados para os tickers selecionados. Verifique os códigos ou tente novamente.")
        return None, None

    # Usa "Adj Close" se disponível, caso contrário, usa "Close"
    prices = prices["Adj Close"] if "Adj Close" in prices else prices["Close"]

    # Ajustar caso apenas um ticker seja selecionado
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        prices.columns = [tickers[0].rstrip(".SA")]

    # Remove o sufixo ".SA" dos nomes das colunas
    prices.columns = prices.columns.str.rstrip(".SA")

    # Adiciona o IBOV ao DataFrame
    ibov_data = yf.download("^BVSP", start=start_date, end=end_date)
    prices['IBOV'] = ibov_data["Adj Close"] if "Adj Close" in ibov_data else ibov_data["Close"]

    return tickers, prices

def calculate_beta(returns, market_returns):
    covariance = np.cov(returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    return beta

def calculate_correlation(prices):
    return prices.corr()

def main_dashboard(tickers, prices):
    weights = np.ones(len(tickers)) / len(tickers)
    prices['portfolio'] = prices.drop("IBOV", axis=1) @ weights
    norm_prices = 100 * prices / prices.iloc[0]
    returns = prices.pct_change()[1:]
    vols = returns.std() * np.sqrt(252)
    rets = (norm_prices.iloc[-1] - 100) / 100

    market_returns = returns["IBOV"]
    betas = {
        t: calculate_beta(returns[t], market_returns) for t in prices.columns if t != "IBOV"
    }

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

@st.cache_data(ttl=300)
def fetch_ohlc(ticker, period, interval):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Se vier MultiIndex (já vi acontecer), achata
    if isinstance(df.columns, pd.MultiIndex):
        # tenta pegar o nível 0 (Open/High/Low/Close/Volume)
        df.columns = df.columns.get_level_values(0)

    # Garante que as colunas existem
    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    # Converte pra numérico (se vier object)
    for col in required + (["Volume"] if "Volume" in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove linhas inválidas
    df = df.dropna(subset=required)

    # Ordena por data
    df = df.sort_index()

    return df


def technical_analysis_dashboard():
    st.subheader("📈 Análise Técnica (Candles)")

    # Lista de tickers (local)
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

    interval = {
        "Diário": "1d",
        "Semanal": "1wk",
        "Mensal": "1mo"
    }[interval_label]

    # =========================
    # DADOS
    # =========================
    df = fetch_ohlc(ticker_yf, period=period, interval=interval)
    with st.expander("Debug (dados baixados)"):
    st.write(df.tail(10))
    st.write("Colunas:", list(df.columns))
    st.write("Tipos:", df.dtypes)


    if df is None or df.empty:
        st.error("Não foi possível carregar dados para este ativo/período.")
        return

    # =========================
    # GRÁFICO DE CANDLES
    # =========================
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index.to_pydatetime(),
            open=df["Open"].values,
            high=df["High"].values,
            low=df["Low"].values,
            close=df["Close"].values,
            name=ticker
        )
    )

    if show_volume and "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                opacity=0.25,
                yaxis="y2"
            )
        )

        fig.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                title="Volume"
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=700,
        title=f"{ticker} • {interval_label} • {period}",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # MÉTRICAS (SEMPRE DENTRO DA FUNÇÃO)
    # =========================
    try:
        close_first = float(df["Close"].iloc[0])
        close_last = float(df["Close"].iloc[-1])
        high_max = float(df["High"].max())
        low_min = float(df["Low"].min())

        ret = (close_last / close_first) - 1
        amp = (high_max / low_min) - 1

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


    """
    Exibe múltiplos financeiros de forma robusta e integrada ao layout.
    """
    st.subheader("Múltiplos Financeiros")

    if not tickers:
        st.warning("Por favor, insira pelo menos um ticker.")
        return

    try:
        # Coleta de dados financeiros
        ticker_data = yq.Ticker(tickers)
        summary_data = ticker_data.summary_detail
        key_stats_data = ticker_data.key_stats

        financial_data = []
        for ticker in tickers:
            ticker_clean = ticker.rstrip(".SA")
            summary = summary_data.get(ticker, {})
            key_stats = key_stats_data.get(ticker, {})

            try:
                # Coletando dados para o cálculo do ROE
                net_income = key_stats.get("netIncomeToCommon", None)  # Lucro líquido
                equity = key_stats.get("totalStockholderEquity", None) #NÃO EXISTE NO SUMMARY E NEM NO KEYSTATS
                book_value_per_share = key_stats.get("bookValue") # VALUE PER SHARE
                shares_outstanding = key_stats.get("sharesOutstanding")
                total_equity = book_value_per_share * shares_outstanding

                roe = (
                    f"{(net_income / total_equity) * 100:.2f}%" if net_income and total_equity else "Dado não disponível"
                )

                financial_data.append({
                    "Ticker": ticker_clean,
                    "P/L": summary.get("trailingPE", "Indisponível"),
                    "P/VP": key_stats.get("priceToBook", "Indisponível"),
                    "LPA": key_stats.get("trailingEps", "Indisponível"),
                    "Margem Bruta": f"{key_stats.get('profitMargins', 0) * 100:.2f}%" if key_stats.get('profitMargins') is not None else "Indisponível",
                    "Margem EBITDA": f"{key_stats.get('enterpriseToEbitda', 0):.2f}" if key_stats.get('enterpriseToEbitda') is not None else "Indisponível",
                    "ROE": roe,  # ROE CALCULADO MANUALMENTE
                    "Dividend Yield": f"{summary.get('dividendYield', 0) * 100:.2f}%" if summary.get('dividendYield') is not None else "Indisponível",
                })
            except Exception as e:
                st.error(f"Erro ao processar dados de {ticker_clean}: {e}")

        # Exibição dos dados financeiros
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


        # Exibição dos dados financeiros
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
    """
    Mostra histórico de dividendos (por ação) em barras ano a ano
    para os tickers selecionados.
    """
    st.subheader("📊 Histórico de Dividendos (ano a ano)")

    if not tickers:
        st.warning("Selecione pelo menos um ticker.")
        return

    # Parâmetro simples pra filtrar anos muito antigos
    min_year = st.slider("Mostrar a partir do ano", 2000, datetime.now().year, 2015)

    annual = {}
    for t in tickers:
        t_clean = t.replace(".SA", "")
        try:
            divs = yf.Ticker(t).dividends  # Série com datas e valores (cash dividend por ação)

            if divs is None or len(divs) == 0:
                annual[t_clean] = pd.Series(dtype=float)
                continue

            divs = divs.copy()
            divs.index = pd.to_datetime(divs.index)
            annual_sum = divs.resample("Y").sum()
            annual_sum.index = annual_sum.index.year
            annual_sum = annual_sum[annual_sum.index >= min_year]

            annual[t_clean] = annual_sum

        except Exception as e:
            st.error(f"Erro ao obter dividendos de {t_clean}: {e}")
            annual[t_clean] = pd.Series(dtype=float)

    if not any(len(s) > 0 for s in annual.values()):
        st.warning("Não encontrei histórico de dividendos para os ativos selecionados.")
        return

    # Junta num DataFrame: linhas = anos, colunas = tickers
    df = pd.DataFrame(annual).fillna(0)
    df.index.name = "Ano"

    st.dataframe(df, use_container_width=True)

    # Gráfico em barras (Plotly) - empilhado para ver soma e composição
    df_plot = df.reset_index().melt(id_vars="Ano", var_name="Ticker", value_name="Dividendos")
    fig = px.bar(
        df_plot,
        x="Ano",
        y="Dividendos",
        color="Ticker",
        barmode="group",  # troque para "stack" se preferir empilhado
        title="Dividendos por ação (soma no ano)",
        template="plotly_dark",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)




def rrg_graph(tickers, prices):
    """
    Gera o gráfico RRG (Relative Rotation Graph) com filtro por setor e/ou tickers selecionados.
    """
    st.subheader("Relative Rotation Graph (RRG)")

    try:
        # 1. Carrega o mapeamento de tickers por setor
        setor_data = pd.read_csv("tickers/tickers_setor.csv", encoding='latin1')  # Usando 'latin1'
        setores = setor_data["Setor"].unique()

        # 2. Adiciona um combo box para seleção de setor
        setor_selecionado = st.selectbox("Selecione o Setor", ["Nenhum"] + list(setores))

        # 3. Filtra os tickers com base no setor selecionado
        if setor_selecionado == "Todos":
            tickers_setor = []  # Nenhum setor selecionado
        else:
            tickers_setor = setor_data[setor_data["Setor"] == setor_selecionado]["Ticker"].tolist()
            tickers_setor = [t + ".SA" for t in tickers_setor]

        # 4. Combina os tickers selecionados manualmente com os do setor (se houver)
        if tickers:
            tickers_filtrados = list(set(tickers + tickers_setor))  # Remove duplicatas
        else:
            tickers_filtrados = tickers_setor

        # 5. Se nenhum ticker foi selecionado (nem manualmente, nem por setor), exibe uma mensagem
        if not tickers_filtrados:
            st.warning("Selecione pelo menos um ativo manualmente ou escolha um setor.")
            return

        # 6. Define o período de análise (últimos 2 anos)
        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(years=2)

        # 7. Baixa os preços dos tickers filtrados
        prices_filtrados = yf.download(tickers_filtrados, start=start_date, end=end_date)

        if prices_filtrados.empty:
            st.error("Não foi possível obter dados para os tickers selecionados.")
            return

        # 8. Filtra tickers com dados válidos
        valid_tickers = [t for t in tickers_filtrados if t in prices_filtrados["Close"].columns]
        if not valid_tickers:
            st.error("Nenhum ticker válido encontrado.")
            return

        # 9. Usa "Close" como alternativa se "Adj Close" não estiver disponível
        if "Adj Close" in prices_filtrados:
            prices_filtrados = prices_filtrados["Adj Close"]
        else:
            prices_filtrados = prices_filtrados["Close"]

        # 10. Ajustar caso apenas um ticker seja selecionado
        if isinstance(prices_filtrados, pd.Series):
            prices_filtrados = prices_filtrados.to_frame()
            prices_filtrados.columns = [valid_tickers[0].rstrip(".SA")]

        # 11. Remove o sufixo ".SA" dos nomes das colunas
        prices_filtrados.columns = prices_filtrados.columns.str.rstrip(".SA")

        # 12. Adiciona o IBOV ao DataFrame
        ibov_data = yf.download("^BVSP", start=start_date, end=end_date)
        prices_filtrados['IBOV'] = ibov_data["Close"]  # Usa "Close" como alternativa

        # 13. Calcula os retornos semanais (para suavizar oscilações diárias)
        weekly_prices = prices_filtrados.resample('W').last()  # Preços no final de cada semana
        weekly_returns = weekly_prices.pct_change().dropna()

        if weekly_returns.empty or len(weekly_returns) < 10:  # Verifica se há dados suficientes
            st.error("Não há dados suficientes para calcular os retornos semanais.")
            return

        # 14. Calcula a força relativa (RS) em relação ao benchmark (IBOV)
        benchmark_returns = weekly_returns["IBOV"]
        relative_strength = weekly_returns.div(benchmark_returns, axis=0) - 1

        # 15. Calcula o momentum (diferença da força relativa em um período)
        lookback_period = 12  # Período de 12 semanas para calcular o momentum
        momentum = relative_strength - relative_strength.shift(lookback_period)

        # 16. Normaliza os dados para facilitar a comparação
        relative_strength_norm = (relative_strength - relative_strength.mean()) / relative_strength.std()
        momentum_norm = (momentum - momentum.mean()) / momentum.std()

        # 17. Verifica se há dados válidos para o gráfico
        if relative_strength_norm.empty or momentum_norm.empty:
            st.error("Não há dados suficientes para gerar o gráfico RRG.")
            return

        # 18. Prepara os dados para o gráfico
        data = pd.DataFrame({
            "Ticker": relative_strength_norm.columns,
            "Relative Strength": relative_strength_norm.iloc[-1].values,
            "Momentum": momentum_norm.iloc[-1].values,
        })

        # 19. Define os quadrantes
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

        # 20. Cores para cada quadrante
        quadrant_colors = {
            "Líderes": "green",
            "Melhorando": "blue",
            "Enfraquecendo": "orange",
            "Defasados": "red",
        }

        # 21. Plota o gráfico RRG usando Plotly
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

        # 22. Adiciona quadrantes e linhas de referência
        fig.add_shape(type="line", x0=0, y0=-2, x1=0, y1=2, line=dict(color="white", dash="dash"))
        fig.add_shape(type="line", x0=-2, y0=0, x1=2, y1=0, line=dict(color="white", dash="dash"))

        # 23. Adiciona anotações para os quadrantes
        fig.add_annotation(x=1, y=1, text="Líderes", showarrow=False, font=dict(color="green", size=14))
        fig.add_annotation(x=-1, y=1, text="Melhorando", showarrow=False, font=dict(color="blue", size=14))
        fig.add_annotation(x=1, y=-1, text="Enfraquecendo", showarrow=False, font=dict(color="orange", size=14))
        fig.add_annotation(x=-1, y=-1, text="Defasados", showarrow=False, font=dict(color="red", size=14))

        # 24. Ajusta o layout
        fig.update_traces(
            marker=dict(size=15, line=dict(width=2, color="DarkSlateGrey")),
            textposition="top center",
        )
        fig.update_layout(
            xaxis_range=[-2, 2],  # Ajuste conforme necessário
            yaxis_range=[-2, 2],  # Ajuste conforme necessário
            showlegend=False,  # Remove a legenda padrão
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo CSV ou gerar o gráfico RRG: {e}")



def get_ibovespa_composition():
    """Obtém a composição atual do Ibovespa com pesos"""
    # Lista completa dos ativos do Ibovespa (atualizada em agosto/2023)
    composition = {
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
        # Adicione outros ativos conforme necessário
    }
    return composition

def get_real_time_prices(tickers):
    """Obtém os preços em tempo real dos ativos com tratamento de erros"""
    tickers_list = [t + ".SA" for t in tickers]
    try:
        data = yf.download(tickers_list, period="1d", progress=False, group_by='ticker')
        
        prices = {}
        for t in tickers_list:
            try:
                if isinstance(data, pd.DataFrame):
                    # Quando há apenas um ticker
                    if "Adj Close" in data:
                        prices[t.replace(".SA", "")] = data["Adj Close"].iloc[-1]
                    else:
                        prices[t.replace(".SA", "")] = data["Close"].iloc[-1]
                else:
                    # Quando há múltiplos tickers
                    if "Adj Close" in data[t]:
                        prices[t.replace(".SA", "")] = data[t]["Adj Close"].iloc[-1]
                    else:
                        prices[t.replace(".SA", "")] = data[t]["Close"].iloc[-1]
            except:
                prices[t.replace(".SA", "")] = None
                
        return prices
    except Exception as e:
        st.error(f"Erro ao baixar cotações: {e}")
        return {t: None for t in tickers}

def ibovespa_map():
    """Mapa do Ibovespa com variação diária e formatação aprimorada"""
    st.subheader("🗺️ Mapa do Ibovespa - Composição por Setor")
    
    try:
        composition = get_ibovespa_composition()
        tickers = [t + ".SA" for t in composition.keys()]
        
        with st.spinner("Obtendo dados em tempo real..."):
            # Baixa dados dos últimos 2 dias para calcular variação
            data = yf.download(tickers, period="2d", group_by="ticker", progress=False)
            
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
                            
                            # Formatação dos valores
                            formatted_price = f"R$ {current_price:.2f}"
                            formatted_weight = f"{composition[ticker]['peso']:.2f}%"
                            formatted_variation = f"{variation:+.2f}%"
                            
                            # Cor baseada na variação
                            color = "green" if variation >= 0 else "red"
                            
                            plot_data.append({
                                "Ticker": ticker,
                                "Setor": composition[ticker]["setor"],
                                "Peso": composition[ticker]["peso"],
                                "Preço": current_price,
                                "Variação": variation,
                                "Texto": (
                                    f"<b style='font-size:16px; text-align:center'>{ticker}</b><br>"
                                    f"<span style='font-size:14px'>{formatted_price}</span><br>"
                                    f"<span style='font-size:12px; color:{color}'>{formatted_variation}</span><br>"
                                    f"<span style='font-size:10px'>{formatted_weight}</span>"
                                )
                            })
                except Exception as e:
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
            
            # Ajustes estéticos finais
            fig.update_traces(
                texttemplate='<b>%{label}</b><br>%{customdata[0]}',
                textfont=dict(
                    family="Arial Black",
                    color="black",
                    size=14
                ),
                textposition="middle center",
                marker=dict(line=dict(width=1, color='DarkSlateGrey'))
            )
            
            fig.update_layout(
                margin=dict(t=50, l=25, r=25, b=25),
                uniformtext=dict(
                    minsize=12,
                    mode='hide'
                ),
                coloraxis_colorbar=dict(
                    title="Variação (%)",
                    tickprefix="%",
                    thickness=15
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Erro ao gerar o mapa: {str(e)}")
        

# Configuração inicial
st.set_page_config(layout="wide")

# Primeiro definimos a seleção de abas
with st.sidebar:
    selected_tab = st.radio(
        "Escolha a visualização",
        ["Dashboard", "Correlação", "Múltiplos", "Dividendos",
         "Análise Técnica", "Notícias", "RRG", "Mapa Ibovespa"]
    )

    if selected_tab in ["Dashboard", "Correlação", "Múltiplos", "Dividendos", "RRG"]:
        tickers, prices = build_sidebar()
    else:
        tickers, prices = None, None

# Título principal
st.title('Renova Invest - Mercado de Capitais')

# Lógica para exibir a aba correta
# Lógica para exibir a aba correta
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

elif selected_tab == "RRG":
    if tickers and prices is not None:
        rrg_graph(tickers, prices)
    else:
        st.warning("Por favor, selecione pelo menos um ticker na barra lateral.")

