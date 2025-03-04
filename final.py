import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ====== THEME ====== #
st.set_page_config(
    page_title="Stock Predictor AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Finance Theme
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
    }
    .sidebar .sidebar-content {
        background: #1c1e26;
        color: white;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #FFD700;
    }
    .css-18e3th9 {
        background-color: #0e1117 !important;
    }
    .css-1d391kg {
        color: #FFD700;
    }
    </style>
    """, unsafe_allow_html=True)

# ====== DATA LOADING ====== #
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

# ====== MACHINE LEARNING ====== #
features = ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'Lag_1', 'Lag_7']
target = 'Close'

X_train, X_test, y_train, y_test = train_test_split(stock_df[features], stock_df[target], test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "stock_price_model.pkl")

# ====== SIDEBAR - MACRO INDICATORS ====== #
st.sidebar.title("📊 Macro Economic Indicators")
inflation = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 3.5, step=0.1)
gdp_growth = st.sidebar.slider("GDP Growth (%)", -5.0, 10.0, 2.0, step=0.1)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 3.0, step=0.1)
unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 4.5, step=0.1)
bond_yield = st.sidebar.slider("10Y Treasury Yield (%)", 0.0, 5.0, 2.5, step=0.1)

macro_factors = {"Inflation": inflation, "GDP Growth": gdp_growth, "Interest Rate": interest_rate, "Unemployment": unemployment, "Bond Yield": bond_yield}

# ====== STOCK SELECTION ====== #
st.sidebar.header("📈 Stock Prediction Settings")
ticker = st.sidebar.selectbox("Select Stock", stock_df['Ticker'].unique())
days_ahead = st.sidebar.slider("Days Ahead to Predict", 1, 30, 7)

selected_df = stock_df[stock_df['Ticker'] == ticker]

# ====== MAIN CONTENT ====== #
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"<h2 style='text-align: center; color: #FFD700;'>📊 {ticker} Stock Analysis</h2>", unsafe_allow_html=True)
    
    # Historical Stock Prices
    fig = px.line(selected_df, x='Date', y='Close', title=f"{ticker} Closing Prices", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Predictions
    st.subheader("🔮 Stock Price Prediction")
    last_row = selected_df.iloc[-1]
    pred_features = np.array([[last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume'],
                               last_row['MA_7'], last_row['MA_30'], last_row['Lag_1'], last_row['Lag_7']]])
    pred_prices = [model.predict(pred_features)[0] for _ in range(days_ahead)]

    dates_future = pd.date_range(start=selected_df['Date'].max(), periods=days_ahead+1)[1:]
    pred_df = pd.DataFrame({'Date': dates_future, 'Predicted Price': pred_prices})
    
    fig_pred = px.line(pred_df, x='Date', y='Predicted Price', title="Predicted Prices", template="plotly_dark")
    st.plotly_chart(fig_pred, use_container_width=True)

with col2:
    st.markdown("<h2 style='text-align: center; color: #FFD700;'>📌 Macro Insights</h2>", unsafe_allow_html=True)
    
    macro_df = pd.DataFrame(macro_factors, index=[0]).T
    macro_df.columns = ["Value"]
    macro_df["Impact"] = ["📉" if v > 3 else "📈" for v in macro_df["Value"]]
    
    st.dataframe(macro_df.style.applymap(lambda x: "color: red;" if x == "📉" else "color: green;", subset=["Impact"]))

# ====== FOOTER ====== #
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: #1c1e26;
        color: #FFD700;
    }
    </style>
    <div class='footer'>
        <p>🚀 Powered by AI & Financial Data | Developed by Daniel Pace</p>
    </div>
    """, unsafe_allow_html=True)
