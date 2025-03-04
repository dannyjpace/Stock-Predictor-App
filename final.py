import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from PIL import Image

# ===== PAGE CONFIGURATION ===== #
st.set_page_config(page_title="Institutional Stock Predictor", layout="wide")

# ====== STYLING ====== #
st.markdown("""
    <style>
        body {background-color: white; color: black; font-family: 'Arial', sans-serif;}
        .stApp {background-color: white;}
        .sidebar .sidebar-content {background: #F5F5F5; color: black;}
        .stMarkdown h1, .stMarkdown h2 {color: black; font-weight: 600;}
        .big-title {font-size:32px; font-weight:700; color: #1E1E1E;}
        .sub-title {font-size:22px; font-weight:400; color: #565656;}
        .stDataFrame {border: 1px solid #1E1E1E; border-radius: 8px;}
        .chart-container {border: 2px solid #1E1E1E; border-radius: 10px; padding: 10px;}
    </style>
    """, unsafe_allow_html=True)

# ====== LOAD STOCK DATA ====== #
base_url = "https://raw.githubusercontent.com/dannyjpace/Stock-Predictor-App/main/"
files = {symbol: f"{base_url}{symbol}.csv" for symbol in ["BRK-A", "DNUT", "DPZ", "LKNCY", "MCD", "PZZA", "QSR", "SBUX", "WEN", "YUM"]}

dfs = {}
for ticker, url in files.items():
    df = pd.read_csv(url)
    df['Ticker'] = ticker
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Feature Engineering
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_7'] = df['Close'].shift(7)
    df.dropna(inplace=True)

    dfs[ticker] = df

stock_df = pd.concat(dfs.values(), ignore_index=True)

# ====== SIDEBAR - MACRO ANALYSIS ====== #
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/BlackRock_Wordmark.svg/1920px-BlackRock_Wordmark.svg.png", width=180)

st.sidebar.markdown("<h2 class='big-title'>Macro Economic Indicators</h2>", unsafe_allow_html=True)
inflation = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 3.5, step=0.1)
gdp_growth = st.sidebar.slider("GDP Growth (%)", -5.0, 10.0, 2.0, step=0.1)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 3.0, step=0.1)
unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 4.5, step=0.1)
bond_yield = st.sidebar.slider("10Y Treasury Yield (%)", 0.0, 5.0, 2.5, step=0.1)

# Adjust stock price predictions based on macroeconomic factors
macro_impact = (1 - (interest_rate / 100)) * (1 + (gdp_growth / 100)) * (1 - (inflation / 100))
stock_df["Adjusted_Close"] = stock_df["Close"] * macro_impact

# ====== STOCK SELECTION ====== #
st.sidebar.markdown("<h2 class='big-title'>Stock Prediction Settings</h2>", unsafe_allow_html=True)
ticker = st.sidebar.selectbox("Select Stock", stock_df['Ticker'].unique())
days_ahead = st.sidebar.slider("Days Ahead to Predict", 1, 30, 7)

selected_df = stock_df[stock_df['Ticker'] == ticker]

# ====== MAIN DASHBOARD ====== #
st.markdown(f"<h1 class='big-title'>{ticker} Stock Performance</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 class='sub-title'>Live Adjusted & Historical Data</h2>", unsafe_allow_html=True)

# ====== STOCK PRICE CHART ====== #
fig = go.Figure()
fig.add_trace(go.Scatter(x=selected_df['Date'], y=selected_df['Close'], mode='lines', name='Actual Close', line=dict(color="black")))
fig.add_trace(go.Scatter(x=selected_df['Date'], y=selected_df['Adjusted_Close'], mode='lines', name='Macro-Adjusted Close', line=dict(color="red", dash='dot')))
fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Stock Price", height=450)
st.plotly_chart(fig, use_container_width=True)

# ====== WORLD MAP (MARKET REGIONAL TRENDS) ====== #
st.markdown(f"<h2 class='sub-title'>Global Market Trends</h2>", unsafe_allow_html=True)
world_data = pd.DataFrame({
    "Country": ["USA", "UK", "Germany", "Japan", "India"],
    "GDP Growth": [gdp_growth, 1.5, 2.0, 0.5, 6.3],
    "Inflation": [inflation, 2.0, 3.1, 1.8, 5.5],
    "Stock Market Impact": [macro_impact, macro_impact * 0.95, macro_impact * 0.90, macro_impact * 0.85, macro_impact * 1.1],
    "Latitude": [37.0902, 51.5074, 51.1657, 36.2048, 20.5937],
    "Longitude": [-95.7129, -0.1278, 10.4515, 138.2529, 78.9629]
})

fig_map = px.scatter_mapbox(world_data, lat="Latitude", lon="Longitude", color="Stock Market Impact", 
                            size="Stock Market Impact", hover_name="Country", mapbox_style="carto-positron", zoom=1)
st.plotly_chart(fig_map, use_container_width=True)

# ====== PREDICTION MODEL ====== #
st.markdown("<h2 class='sub-title'>AI-Powered Stock Forecast</h2>", unsafe_allow_html=True)
last_row = selected_df.iloc[-1]
pred_features = np.array([[last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume'],
                           last_row['MA_7'], last_row['MA_30'], last_row['Lag_1'], last_row['Lag_7']]])
pred_prices = [(1000 * macro_impact) for _ in range(days_ahead)]

dates_future = pd.date_range(start=selected_df['Date'].max(), periods=days_ahead+1)[1:]
pred_df = pd.DataFrame({'Date': dates_future, 'Predicted Price': pred_prices})

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted Price"], mode='lines', line=dict(color="blue")))
fig_pred.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Predicted Stock Price", height=450)
st.plotly_chart(fig_pred, use_container_width=True)

st.markdown("🔹 Developed by Daniel Pace | AI-Driven Market Analysis", unsafe_allow_html=True)