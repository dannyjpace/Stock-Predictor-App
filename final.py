import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from io import StringIO
import openai
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# ---- LLM API KEY ----
openai.api_key = st.secrets["OPENAI_API_KEY"]  # put in .streamlit/secrets.toml

# ---- GITHUB CSV FILES ----
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dannyjpace/Stock-Predictor-App/main/"
files = {
    "BRK-A": "BRK-A.csv", "DNUT": "DNUT.csv", "DPZ": "DPZ.csv", "LKNCY": "LKNCY.csv",
    "MCD": "MCD.csv", "PZZA": "PZZA.csv", "QSR": "QSR.csv", "SBUX": "SBUX.csv",
    "WEN": "WEN.csv", "YUM": "YUM.csv",
}

def clean_stock_data(df):
    df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(window=7, min_periods=1).mean())
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        mean, std = df[col].mean(), df[col].std()
        z_scores = (df[col] - mean) / std
        outliers = df[np.abs(z_scores) > 3].index
        df.loc[outliers, col] = df[col].rolling(window=7, min_periods=1).median()
        df[col] = df[col].fillna(df[col].rolling(window=7, min_periods=1).median())
    df.drop_duplicates(inplace=True)
    if 'Date' in df.columns:
        df.drop_duplicates(subset='Date', keep='first', inplace=True)
    return df

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

stock_df = load_data()

# ---- SIDEBAR ----
st.sidebar.title("Stock Market Predictor")
ticker = st.sidebar.selectbox("Select Stock", stock_df["Ticker"].unique())
days_ahead = st.sidebar.slider("Days Ahead to Predict", 1, 30, 7)
use_advanced_model = st.sidebar.toggle("Use Advanced AI Model")

# ---- MACRO INPUT ----
st.sidebar.subheader("Macro Indicators")
inflation = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 3.5)
gdp_growth = st.sidebar.slider("GDP Growth (%)", -5.0, 10.0, 2.0)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 3.0)
unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 6.5)
treasury_yield = st.sidebar.slider("10Y Treasury Yield (%)", 0.0, 5.0, 2.5)

macro_impact = (
    (interest_rate * -0.05) +
    (gdp_growth * 0.08) +
    (inflation * -0.03) +
    (unemployment * -0.02) +
    (treasury_yield * -0.04)
)

selected_df = stock_df[stock_df["Ticker"] == ticker]

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
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    return model, scaler

model, scaler = train_lstm_model(selected_df)

def predict_future_prices(model, df, scaler, days):
    last_data = df[["Close"]].values[-10:]
    last_data_scaled = scaler.transform(last_data)
    predictions = []
    for _ in range(days):
        input_data = last_data_scaled[-10:].reshape(1, 10, 1)
        predicted_price = model.predict(input_data, verbose=0)[0][0]
        predictions.append(predicted_price)
        last_data_scaled = np.append(last_data_scaled[1:], [[predicted_price]], axis=0)
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predicted_prices * (1 + macro_impact)

pred_prices = predict_future_prices(model, selected_df, scaler, days_ahead)
dates_future = pd.date_range(start=selected_df["Date"].max(), periods=days_ahead + 1)[1:]
pred_df = pd.DataFrame({"Date": dates_future, "Predicted Price": pred_prices})
full_df = pd.concat([selected_df[["Date", "Close"]], pred_df.rename(columns={"Predicted Price": "Close"})])

st.subheader("Latest Stock Price")
latest_price = selected_df["Close"].iloc[-1]
st.metric(label=f"{ticker} Latest Price", value=f"${latest_price:,.2f}")

fig = px.line(full_df, x="Date", y="Close", title=f"{ticker} Price Forecast", template="plotly_white")
fig.add_vline(x=selected_df["Date"].max(), line_dash="dash", line_color="red")
st.plotly_chart(fig)

st.subheader("Volatility & Momentum")
st.write(f"Standard Deviation: {selected_df['Close'].std():.2f}")
st.write(f"7-Day MA: {selected_df['MA_7'].iloc[-1]:.2f}")
st.write(f"Momentum: {selected_df['Close'].pct_change().mean():.4f}")

st.subheader("Forecasted Prices")
st.dataframe(pred_df)

# ---- LLM Insight ----
if st.checkbox("Generate Market Insight with LLM"):
    trend = "rising" if pred_prices[-1] > pred_prices[0] else "falling" if pred_prices[-1] < pred_prices[0] else "stable"

    prompt = f"""
    Based on the macroeconomic inputs:
    - Inflation Rate: {inflation}%
    - GDP Growth: {gdp_growth}%
    - Interest Rate: {interest_rate}%
    - Unemployment: {unemployment}%
    - 10Y Treasury Yield: {treasury_yield}%

    And the predicted stock trend for {ticker} over the next {days_ahead} days, which is {trend},
    generate a short 2-3 sentence market commentary for a business analyst.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    st.subheader("LLM Market Insight")
    st.info(response["choices"][0]["message"]["content"])
