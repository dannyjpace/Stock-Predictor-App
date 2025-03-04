import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf
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

# ---- DATA CLEANING FUNCTION ----
def clean_stock_data(df):
    df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(window=7, min_periods=1).mean())

    # Handling outliers
    for column in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        mean, std = df[column].mean(), df[column].std()
        z_scores = (df[column] - mean) / std
        outliers = df[np.abs(z_scores) > 3].index
        df.loc[outliers, column] = df[column].rolling(window=7, min_periods=1).median()
        df[column] = df[column].fillna(df[column].rolling(window=7, min_periods=1).median())

    # Remove duplicate dates
    df.drop_duplicates(inplace=True)
    if 'Date' in df.columns:
        df.drop_duplicates(subset='Date', keep='first', inplace=True)

    return df

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
            df = clean_stock_data(df)
            df["MA_7"] = df["Close"].rolling(window=7).mean()
            df["MA_30"] = df["Close"].rolling(window=30).mean()
            df["Lag_1"] = df["Close"].shift(1)
            df["Lag_7"] = df["Close"].shift(7)
            df.dropna(inplace=True)
            dfs[ticker] = df
    return pd.concat(dfs.values(), ignore_index=True)

# ---- LOAD DATASET ----
stock_df = load_data()

# ---- SIDEBAR ----
st.sidebar.title("Stock Market Predictor")
st.sidebar.subheader("Stock Prediction Settings")

ticker = st.sidebar.selectbox("Select Stock", stock_df["Ticker"].unique())
days_ahead = st.sidebar.slider("Days Ahead to Predict", 1, 30, 7)

use_advanced_model = st.sidebar.toggle("Use Advanced AI Model")

# ---- MACROECONOMIC FACTORS ----
st.sidebar.subheader("Macro Indicators")
inflation = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 3.5)
gdp_growth = st.sidebar.slider("GDP Growth (%)", -5.0, 10.0, 2.0)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 3.0)
unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 6.5)
treasury_yield = st.sidebar.slider("10Y Treasury Yield (%)", 0.0, 5.0, 2.5)

# ---- APPLY MACRO FACTORS TO STOCK PRICE ----
macro_impact = (
    (interest_rate * -0.05) +  # Higher interest rates -> Lower stock price
    (gdp_growth * 0.08) +      # Higher GDP growth -> Higher stock price
    (inflation * -0.03) +      # Higher inflation -> Lower stock price
    (unemployment * -0.02) +   # Higher unemployment -> Lower stock price
    (treasury_yield * -0.04)   # Higher bond yields -> Lower stock price
)

# ---- FILTER DATASET FOR SELECTED STOCK ----
selected_df = stock_df[stock_df["Ticker"] == ticker]

# ---- LSTM MODEL FOR PREDICTION ----
def train_lstm_model(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[["Close"]])

    X, y = [], []
    lookback = 10
    for i in range(len(df_scaled) - lookback):
        X.append(df_scaled[i : i + lookback])
        y.append(df_scaled[i + lookback])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

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

# ---- PREDICT FUTURE PRICES ----
def predict_future_prices(model, df, scaler, days):
    last_data = df[["Close"]].values[-10:]
    last_data_scaled = scaler.transform(last_data)

    predictions = []
    for _ in range(days):
        input_data = last_data_scaled[-10:].reshape(1, 10, 1)
        predicted_price = model.predict(input_data)[0][0]
        predictions.append(predicted_price)
        last_data_scaled = np.append(last_data_scaled[1:], [[predicted_price]], axis=0)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predicted_prices * (1 + macro_impact)  # Apply macroeconomic impact

# ---- DISPLAY PREDICTIONS ----
pred_prices = predict_future_prices(model, selected_df, scaler, days_ahead)
dates_future = pd.date_range(start=selected_df["Date"].max(), periods=days_ahead + 1)[1:]
pred_df = pd.DataFrame({"Date": dates_future, "Predicted Price": pred_prices})

# ---- COMBINE HISTORICAL AND PREDICTED DATA ----
full_df = pd.concat([selected_df[["Date", "Close"]], pred_df.rename(columns={"Predicted Price": "Close"})])

# ---- DISPLAY LATEST STOCK PRICE ----
st.subheader("Latest Stock Price")
latest_price = selected_df["Close"].iloc[-1]  # Get the most recent stock price
st.metric(label=f"{ticker} Latest Price", value=f"${latest_price:,.2f}")

# ---- DISPLAY COMBINED GRAPH ----
fig = px.line(full_df, x="Date", y="Close", title=f"{ticker} Price Forecast", template="plotly_white")
fig.add_vline(x=selected_df["Date"].max(), line_dash="dash", line_color="red")  # Mark the prediction start date
st.plotly_chart(fig)

# ---- ADDITIONAL METRICS ----
st.subheader("Stock Volatility & Momentum")
st.write(f"Standard Deviation: {selected_df['Close'].std():.2f}")
st.write(f"Moving Average (7-day): {selected_df['MA_7'].iloc[-1]:.2f}")
st.write(f"Momentum: {selected_df['Close'].pct_change().mean():.4f}")

st.subheader("Forecasted Prices")
st.dataframe(pred_df)
