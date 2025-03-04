import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# ====================== UI ENHANCEMENTS ====================== #
st.set_page_config(page_title="Stock Predictor Pro", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        body { background-color: #F4F4F4; }
        .css-1aumxhk, .css-1dp5vir { background-color: white !important; }
        .stSlider { font-size: 18px; }
        .stButton { border-radius: 10px; }
        .css-18e3th9 { padding-top: 50px; }
    </style>
""", unsafe_allow_html=True)

# ====================== DATA LOADING ====================== #
st.sidebar.title("📊 Stock Market Predictor")

# Define raw URL for GitHub
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

# Load datasets
dfs = {}
for ticker, url in files.items():
    df = pd.read_csv(url)
    df["Ticker"] = ticker
    df["Date"] = pd.to_datetime(df["Date"])

    # Feature Engineering
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_30"] = df["Close"].rolling(window=30).mean()
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_7"] = df["Close"].shift(7)

    # Remove Spikes using IQR
    def remove_spikes(df, column):
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[column] = np.where(
            (df[column] < lower) | (df[column] > upper),
            df[column].rolling(window=7, min_periods=1).median(),
            df[column],
        )
        return df

    for col in ["Open", "High", "Low", "Close"]:
        df = remove_spikes(df, col)

    df["Close"] = df["Close"].replace(0, df["Close"].rolling(window=5, min_periods=1).mean())

    dfs[ticker] = df

# Combine cleaned data
stock_df = pd.concat(dfs.values(), ignore_index=True)

# ====================== MACRO FACTORS ====================== #
st.sidebar.header("Macro Economic Factors")
inflation = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 2.5)
gdp_growth = st.sidebar.slider("GDP Growth (%)", -5.0, 10.0, 2.0)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 3.0)
unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 5.0)
bond_yield = st.sidebar.slider("10Y Treasury Yield (%)", 0.0, 5.0, 2.5)

macro_factors = np.array([[inflation, gdp_growth, interest_rate, unemployment, bond_yield]])

# ====================== STOCK SELECTION ====================== #
st.sidebar.header("Stock Selection")
ticker = st.sidebar.selectbox("Choose a Stock", stock_df["Ticker"].unique())

# ====================== DISPLAY HISTORICAL DATA ====================== #
selected_df = stock_df[stock_df["Ticker"] == ticker]

st.markdown(f"## {ticker} Stock Analysis")

fig = px.line(selected_df, x="Date", y="Close", title=f"{ticker} Closing Prices", template="plotly_dark")
st.plotly_chart(fig)

# ====================== TRAIN A DEEP LEARNING MODEL ====================== #
# Data Preprocessing
features = ["Open", "High", "Low", "Volume", "MA_7", "MA_30", "Lag_1", "Lag_7"]
target = "Close"

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(selected_df[features])

X, y = [], []
sequence_length = 30  # Using last 30 days to predict next day
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i : i + sequence_length])
    y.append(scaled_data[i + sequence_length, 0])  # Predicting "Open" price

X, y = np.array(X), np.array(y)

# Split into training & testing
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

# Predict Future Stock Prices
days_ahead = st.sidebar.slider("Predict Days Ahead", 1, 30, 7)

future_predictions = []
last_sequence = X[-1]  # Last known data

for _ in range(days_ahead):
    pred_price = model.predict(last_sequence.reshape(1, sequence_length, X.shape[2]))[0, 0]
    future_predictions.append(pred_price)
    new_entry = np.append(last_sequence[1:], [[pred_price] + list(macro_factors[0])], axis=0)
    last_sequence = new_entry

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# ====================== DISPLAY PREDICTIONS ====================== #
st.markdown(f"## {ticker} Stock Price Prediction")
future_dates = pd.date_range(start=selected_df["Date"].max(), periods=days_ahead+1)[1:]
pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})

fig_pred = px.line(pred_df, x="Date", y="Predicted Price", title=f"{ticker} Predicted Prices", template="plotly_dark")
st.plotly_chart(fig_pred)

st.dataframe(pred_df)

# ====================== FOOTER ====================== #
st.markdown("**Powered by AI & Financial Data | Developed by Daniel Pace**")
