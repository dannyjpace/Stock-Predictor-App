import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ====== PAGE CONFIGURATION ====== #
st.set_page_config(
    page_title="Stock Predictor AI",
    page_icon="💹",
    layout="wide"
)

# ====== CUSTOM CSS FOR PROFESSIONAL LOOK ====== #
st.markdown("""
    <style>
        body {background-color: #0e1117; color: white;}
        .stApp {background-color: #0e1117;}
        .sidebar .sidebar-content {background: #11141b; color: white;}
        .css-18e3th9 {background-color: #0e1117 !important;}
        .css-1d391kg {color: #FFD700;}
        .big-font {font-size:24px !important; font-weight:bold; color: #FFD700;}
        .stMarkdown h1, .stMarkdown h2 {color: #FFD700;}
        .stDataFrame {border: 1px solid #FFD700; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

# ====== LOAD DATA FROM GITHUB ====== #
base_url = "https://raw.githubusercontent.com/dannyjpace/Stock-Predictor-App/main/"

files = {
    "BRK-A": base_url + "BRK-A.csv",
    "DNUT": base_url + "DNUT.csv",
    "DPZ": base_url + "DPZ.csv",
    "LKNCY": base_url + "LKNCY.csv",
    "MCD": base_url + "MCD.csv",
    "PZZA": base_url + "PZZA.csv",
    "QSR": base_url + "QSR.csv",
    "SBUX": base_url + "SBUX.csv",
    "WEN": base_url + "WEN.csv",
    "YUM": base_url + "YUM.csv",
}

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
st.sidebar.header("📊 Macro Economic Factors")
inflation = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 3.5, step=0.1)
gdp_growth = st.sidebar.slider("GDP Growth (%)", -5.0, 10.0, 2.0, step=0.1)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 3.0, step=0.1)
unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 4.5, step=0.1)
bond_yield = st.sidebar.slider("10Y Treasury Yield (%)", 0.0, 5.0, 2.5, step=0.1)

# Adjust stock price predictions based on macro factors
macro_impact = (1 - (interest_rate / 100)) * (1 + (gdp_growth / 100)) * (1 - (inflation / 100))
stock_df["Adjusted_Close"] = stock_df["Close"] * macro_impact

# ====== STOCK SELECTION ====== #
st.sidebar.header("📈 Stock Prediction Settings")
ticker = st.sidebar.selectbox("Select Stock", stock_df['Ticker'].unique())
days_ahead = st.sidebar.slider("Days Ahead to Predict", 1, 30, 7)

selected_df = stock_df[stock_df['Ticker'] == ticker]

# ====== MAIN CONTENT ====== #
st.markdown(f"<h1 class='big-font'>📊 {ticker} Stock Analysis</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 style='color: #FFD700;'>📈 Historical & Adjusted Prices</h2>", unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=selected_df['Date'], y=selected_df['Close'], mode='lines', name='Actual Close', line=dict(color="cyan")))
fig.add_trace(go.Scatter(x=selected_df['Date'], y=selected_df['Adjusted_Close'], mode='lines', name='Macro-Adjusted Close', line=dict(color="red", dash='dot')))
fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Stock Price")
st.plotly_chart(fig, use_container_width=True)

# ====== STOCK PRICE PREDICTION ====== #
st.subheader("🔮 Predict Future Stock Prices")
last_row = selected_df.iloc[-1]
pred_features = np.array([[last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume'],
                           last_row['MA_7'], last_row['MA_30'], last_row['Lag_1'], last_row['Lag_7']]])
pred_prices = [(model.predict(pred_features)[0] * macro_impact) for _ in range(days_ahead)]

dates_future = pd.date_range(start=selected_df['Date'].max(), periods=days_ahead+1)[1:]
pred_df = pd.DataFrame({'Date': dates_future, 'Predicted Price': pred_prices})

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted Price"], mode='lines', name="Predicted Prices", line=dict(color="gold")))
fig_pred.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Predicted Stock Price")
st.plotly_chart(fig_pred, use_container_width=True)

# ====== FOOTER ====== #
st.markdown("""
    <div style='text-align: center; padding: 15px; font-size: 16px; color: #FFD700;'>
        🚀 Powered by AI & Financial Data | Developed by Daniel Pace
    </div>
    """, unsafe_allow_html=True)
