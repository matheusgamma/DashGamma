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

def build_sidebar():
    # Carrega a imagem do repositório (caminho relativo)
    st.image("images/Gamma-XP.png")  # Certifique-se de que a imagem está na pasta "images" do repositório

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



def cointegracao(tickers, prices):
    st.subheader("Cointegração - Long & Short")
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

        colA, colB, colC, colD = c.columns([4, 6, 6, 6])
        if t == "portfolio":
            colA.image("images\pie-chart-dollar-svgrepo-com.svg",
                       use_container_width=True)
        elif t == "IBOV":
            colA.image("\images\pie-chart-svgrepo-com.svg",
                       use_container_width=True)
        else:
            colA.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{t}.png', width=85,
                       use_container_width=True)

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


def screening_alerts():
    """
    Verifica setups gráficos (9.1, 9.2, 9.3, 9.4) para todos os tickers do arquivo e exibe alertas.
    """
    st.subheader("Screening Alerts - Setups 9.1, 9.2, 9.3 e 9.4 (Timeframe Diário)")

    # Carrega a lista de tickers do arquivo
    try:
        # Lê o arquivo CSV com um separador de vírgula e sem cabeçalho
        ticker_list = pd.read_csv("tickers/tickers_ibra.csv", header=None, sep=",")
        tickers = [t + ".SA" for t in ticker_list[1]]  # Usa a segunda coluna (tickers)
    except Exception as e:
        st.error(f"Erro ao carregar a lista de tickers: {e}")
        return

    # Baixa os dados de preços para todos os tickers (em lotes para evitar timeout)
    prices = pd.DataFrame()
    batch_size = 10  # Número de tickers por lote
    num_batches = (len(tickers) // batch_size) + 1

    with st.spinner("Baixando dados de preços..."):
        for i in range(num_batches):
            batch = tickers[i * batch_size : (i + 1) * batch_size]
            if not batch:
                continue
            try:
                # Baixa os dados diários
                batch_data = yf.download(batch, period="6mo", interval="1d", progress=False)
                if "Adj Close" in batch_data:
                    batch_prices = batch_data["Adj Close"]
                else:
                    batch_prices = batch_data["Close"]
                prices = pd.concat([prices, batch_prices], axis=1)
            except Exception as e:
                st.warning(f"Erro ao baixar dados para o lote {i + 1}: {e}")

    if prices.empty:
        st.error("Não foi possível obter dados para os tickers.")
        return

    # Remove o sufixo ".SA" dos nomes das colunas
    prices.columns = prices.columns.str.rstrip(".SA")

    # Dicionário para armazenar os alertas
    alerts = []

    # Loop através de cada ticker
    with st.spinner("Analisando setups gráficos..."):
        for ticker in prices.columns:
            data = prices[ticker].dropna()

            if len(data) < 20:  # Verifica se há dados suficientes
                continue

            # Calcula a MME de 9 períodos
            mme_9 = data.rolling(window=9).mean()

            # Preço atual e fechamento anterior
            current_close = data.iloc[-1]
            previous_close = data.iloc[-2]
            current_mme = mme_9.iloc[-1]
            previous_mme = mme_9.iloc[-2]

            # Verifica os setups
            if current_close > current_mme and previous_close <= previous_mme:  # Setup 9.1 (Compra)
                alerts.append({
                    "Ticker": ticker,
                    "Setup": "9.1 (Compra)",
                    "Descrição": f"Preço fechou acima da MME de 9 períodos.",
                    "Data": data.index[-1].strftime("%Y-%m-%d"),
                    "Preço": current_close,
                    "MME": current_mme
                })
            elif current_close > current_mme and previous_close < previous_mme:  # Setup 9.2 (Compra)
                alerts.append({
                    "Ticker": ticker,
                    "Setup": "9.2 (Compra)",
                    "Descrição": f"Preço fechou acima da MME de 9 períodos após estar abaixo.",
                    "Data": data.index[-1].strftime("%Y-%m-%d"),
                    "Preço": current_close,
                    "MME": current_mme
                })
            elif current_close < current_mme and previous_close >= previous_mme:  # Setup 9.3 (Venda)
                alerts.append({
                    "Ticker": ticker,
                    "Setup": "9.3 (Venda)",
                    "Descrição": f"Preço fechou abaixo da MME de 9 períodos.",
                    "Data": data.index[-1].strftime("%Y-%m-%d"),
                    "Preço": current_close,
                    "MME": current_mme
                })
            elif current_close < current_mme and previous_close > previous_mme:  # Setup 9.4 (Venda)
                alerts.append({
                    "Ticker": ticker,
                    "Setup": "9.4 (Venda)",
                    "Descrição": f"Preço fechou abaixo da MME de 9 períodos após estar acima.",
                    "Data": data.index[-1].strftime("%Y-%m-%d"),
                    "Preço": current_close,
                    "MME": current_mme
                })

    # Exibe os alertas
    if alerts:
        st.write("### Alertas de Setups Gráficos")
        for alert in alerts:
            with st.container():
                st.write(f"**{alert['Ticker']}** - {alert['Setup']}")
                st.write(f"**Descrição**: {alert['Descrição']}")
                st.write(f"**Data**: {alert['Data']}")
                st.write(f"**Preço**: {alert['Preço']:.2f}")
                st.write(f"**MME (9)**: {alert['MME']:.2f}")
                st.write("---")

                # Adiciona um gráfico interativo
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Preço'))
                fig.add_trace(go.Scatter(x=mme_9.index, y=mme_9, mode='lines', name='MME (9)'))
                fig.update_layout(
                    title=f"{alert['Ticker']} - Preço vs MME (9)",
                    xaxis_title="Data",
                    yaxis_title="Preço",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum setup gráfico detectado nos tickers.")


# Configuração inicial
st.set_page_config(layout="wide")
with st.sidebar:
    tickers, prices = build_sidebar()
    selected_tab = st.radio("Escolha a visualização", ["Dashboard", "Correlação", "Múltiplos", "RRG", "Cointegração - L&S", "Screening Alerts"])

st.title('Gamma Capital - Mercado de Capitais')
if selected_tab == "Screening Alerts":
    screening_alerts()  # Chama a função independente da seleção de tickers
elif tickers and prices is not None:
    if selected_tab == "Dashboard":
        main_dashboard(tickers, prices)
    elif selected_tab == "Correlação":
        correlation_dashboard(prices)
    elif selected_tab == "Múltiplos":
        multiples_dashboard(tickers)
    elif selected_tab == "Cointegração - L&S":
        cointegracao(tickers, prices)
    elif selected_tab == "RRG":
        rrg_graph(tickers, prices)

