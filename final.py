import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ====== PAGE CONFIGURATION ====== #
st.set_page_config(
    page_title="Stock Market AI",
    layout="wide"
)

# ====== CUSTOM THEME (J.P. Morgan / Goldman Sachs Level) ====== #
st.markdown("""
    <style>
        body {background-color: #121212; color: white; font-family: 'Inter', sans-serif;}
        .stApp {background-color: #121212;}
        .sidebar .sidebar-content {background: #1b1f24; color: white;}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {color: #E5E5E5; font-weight: 600;}
        .stDataFrame {border: 1px solid #E5E5E5; border-radius: 8px;}
        .reportview-container {background: #121212;}
        .big-title {font-size:28px; font-weight:600; color: #E5E5E5;}
        .sub-title {font-size:20px; font-weight:400; color: #b0b3b8;}
        .highlight {color: #00c6ff; font-weight: bold;}
        .stPlotlyChart {border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

# ====== LOAD DATA FROM GITHUB ====== #
base_url = "https://raw.githubusercontent.com/dannyjpace/Stock-Predictor-App/main/"

files = {symbol: f"{base_url}{symbol}.csv" for symbol in [
    "BRK-A", "DNUT", "DPZ", "LKNCY", "MCD", "PZZA", "QSR", "SBUX", "WEN", "YUM"
]}

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

# ====== MACHINE LEARNING MODEL ====== #
features = ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'Lag_1', 'Lag_7']
target = 'Close'

X_train, X_test, y_train, y_test = train_test_split(stock_df[features], stock_df[target], test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "stock_price_model.pkl")

# ====== SIDEBAR - MACRO IMPACT ANALYSIS ====== #
st.sidebar.markdown("<h2 class='big-title'>Macro Economic Factors</h2>", unsafe_allow_html=True)

inflation = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 3.5, step=0.1)
gdp_growth = st.sidebar.slider("GDP Growth (%)", -5.0, 10.0, 2.0, step=0.1)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 3.0, step=0.1)
unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 4.5, step=0.1)
bond_yield = st.sidebar.slider("10Y Treasury Yield (%)", 0.0, 5.0, 2.5, step=0.1)

# Adjust stock price predictions based on macro factors
macro_impact = (1 - (interest_rate / 100)) * (1 + (gdp_growth / 100)) * (1 - (inflation / 100))
stock_df["Adjusted_Close"] = stock_df["Close"] * macro_impact

# ====== STOCK SELECTION ====== #
st.sidebar.markdown("<h2 class='big-title'>Stock Prediction Settings</h2>", unsafe_allow_html=True)
ticker = st.sidebar.selectbox("Select Stock", stock_df['Ticker'].unique())
days_ahead = st.sidebar.slider("Days Ahead to Predict", 1, 30, 7)

selected_df = stock_df[stock_df['Ticker'] == ticker]

# ====== MAIN CONTENT ====== #
st.markdown(f"<h1 class='big-title'>{ticker} Stock Analysis</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 class='sub-title'>Historical & Adjusted Prices</h2>", unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=selected_df['Date'], y=selected_df['Close'], mode='lines', name='Actual Close', line=dict(color="lightgray")))
fig.add_trace(go.Scatter(x=selected_df['Date'], y=selected_df['Adjusted_Close'], mode='lines', name='Macro-Adjusted Close', line=dict(color="dodgerblue", dash='dot')))
fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Stock Price", height=500)
st.plotly_chart(fig, use_container_width=True)

# ====== STOCK PRICE PREDICTION ====== #
st.markdown("<h2 class='sub-title'>Stock Price Prediction</h2>", unsafe_allow_html=True)
last_row = selected_df.iloc[-1]
pred_features = np.array([[last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume'],
                           last_row['MA_7'], last_row['MA_30'], last_row['Lag_1'], last_row['Lag_7']]])
pred_prices = [(model.predict(pred_features)[0] * macro_impact) for _ in range(days_ahead)]

dates_future = pd.date_range(start=selected_df['Date'].max(), periods=days_ahead+1)[1:]
pred_df = pd.DataFrame({'Date': dates_future, 'Predicted Price': pred_prices})

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted Price"], mode='lines', name="Predicted Prices", line=dict(color="cyan")))
fig_pred.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Predicted Stock Price", height=500)
st.plotly_chart(fig_pred, use_container_width=True)

# ====== FOOTER ====== #
st.markdown("""
    <div style='text-align: center; padding: 15px; font-size: 14px; color: #b0b3b8;'>
        Market Data AI | Developed by Daniel Pace | Institutional-Grade Predictions
    </div>
    """, unsafe_allow_html=True)
