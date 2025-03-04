import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
from io import StringIO

# ---- GLOBAL VARIABLES ----
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dannyjpace/Stock-Predictor-App/main/"

# ---- STOCK FILES ----
files = {
    "BRK-A": "BRK-A.csv",
    "DNUT": "DNUT.csv",
    "DPZ": "DPZ.csv",
    "LKNCY": "LKNCY.csv",
    "MCD": "MCD.csv",
    "PZZA": "PZZA.csv",
    "QSR": "QSR.csv",
    "SBUX": "SBUX.csv",
    "WEN": "WEN.csv",
    "YUM": "YUM.csv",
}

# ---- LOAD DATA FUNCTION ----
@st.cache_data
def load_data():
    dfs = {}
    for ticker, filename in files.items():
        url = GITHUB_RAW_BASE + filename
        response = requests.get(url)
        
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            df["Ticker"] = ticker
            df["Date"] = pd.to_datetime(df["Date"])
            
            # Fix "Close" price replacement issue
            df.loc[df["Close"] == 0, "Close"] = df["Close"].rolling(window=5, min_periods=1).mean()
            
            # Feature Engineering
            df["MA_7"] = df["Close"].rolling(window=7).mean()
            df["MA_30"] = df["Close"].rolling(window=30).mean()
            df["Lag_1"] = df["Close"].shift(1)
            df["Lag_7"] = df["Close"].shift(7)

            # Drop NaNs
            df.dropna(inplace=True)
            dfs[ticker] = df
        else:
            print(f"Failed to load data for {ticker}")

    return pd.concat(dfs.values(), ignore_index=True)

# ---- LOAD DATASET ----
stock_df = load_data()

# ---- SIDEBAR ----
st.sidebar.title("Stock Market Predictor")
st.sidebar.subheader("📊 Stock Prediction Settings")

ticker = st.sidebar.selectbox("Select Stock", stock_df["Ticker"].unique())
days_ahead = st.sidebar.slider("Days Ahead to Predict", 1, 30, 7)

# ---- MACROECONOMIC FACTORS ----
st.sidebar.subheader("📈 Macro Indicators")
inflation = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 3.5)
gdp_growth = st.sidebar.slider("GDP Growth (%)", -5.0, 10.0, 2.0)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 3.0)
unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 6.5)
treasury_yield = st.sidebar.slider("10Y Treasury Yield (%)", 0.0, 5.0, 2.5)

# ---- FILTER DATASET FOR SELECTED STOCK ----
selected_df = stock_df[stock_df["Ticker"] == ticker]

# ---- DISPLAY HISTORICAL DATA ----
st.subheader(f"📌 {ticker} Stock Analysis")
fig = px.line(selected_df, x="Date", y="Close", title=f"{ticker} Closing Prices", template="plotly_white")
st.plotly_chart(fig)

# ---- LSTM MODEL FOR PREDICTION ----
def train_lstm_model(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[["Close"]])

    # Create sequences for LSTM
    X, y = [], []
    lookback = 10
    for i in range(len(df_scaled) - lookback):
        X.append(df_scaled[i : i + lookback])
        y.append(df_scaled[i + lookback])

    X, y = np.array(X), np.array(y)

    # Split into train/test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Define LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    return model, scaler

# ---- TRAIN MODEL ----
model, scaler = train_lstm_model(selected_df)

# ---- MAKE PREDICTIONS ----
def predict_future_prices(model, df, scaler, days):
    last_data = df[["Close"]].values[-10:]
    last_data_scaled = scaler.transform(last_data)

    predictions = []
    for _ in range(days):
        input_data = last_data_scaled[-10:].reshape(1, 10, 1)
        predicted_price = model.predict(input_data)[0][0]
        predictions.append(predicted_price)

        # Append the predicted value for next input
        last_data_scaled = np.append(last_data_scaled[1:], [[predicted_price]], axis=0)

    # Convert back to original scale
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    return predicted_prices

# ---- PREDICTED PRICES ----
pred_prices = predict_future_prices(model, selected_df, scaler, days_ahead)

# ---- DISPLAY PREDICTIONS ----
dates_future = pd.date_range(start=selected_df["Date"].max(), periods=days_ahead + 1)[1:]
pred_df = pd.DataFrame({"Date": dates_future, "Predicted Price": pred_prices})

fig_pred = px.line(pred_df, x="Date", y="Predicted Price", title="Predicted Prices", template="plotly_white")

# Formatting the x-axis for cleaner display
fig_pred.update_xaxes(title_text="Date", tickformat="%b %d, %Y")
fig_pred.update_yaxes(title_text="Stock Price")

st.plotly_chart(fig_pred)

st.subheader("🔍 Forecasted Prices")
st.dataframe(pred_df)

# ---- MACRO INSIGHTS TABLE ----
macro_df = pd.DataFrame({
    "Indicator": ["Inflation", "GDP Growth", "Interest Rate", "Unemployment", "Bond Yield"],
    "Value": [inflation, gdp_growth, interest_rate, unemployment, treasury_yield]
})
st.subheader("📌 Macro Insights")
st.table(macro_df)
