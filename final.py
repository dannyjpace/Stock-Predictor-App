import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import streamlit as st
import plotly.express as px

# ✅ Use GitHub raw URLs to load the data
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

# ✅ Load datasets
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

# ✅ Combine all DataFrames into one
stock_df = pd.concat(dfs.values(), ignore_index=True)

# ✅ Train the Model
features = ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'Lag_1', 'Lag_7']
target = 'Close'

X_train, X_test, y_train, y_test = train_test_split(stock_df[features], stock_df[target], test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save the Model
joblib.dump(model, "stock_price_model.pkl")

# ✅ Streamlit App
st.sidebar.header("Stock Price Prediction")
ticker = st.sidebar.selectbox("Select Stock", stock_df['Ticker'].unique())
days_ahead = st.sidebar.slider("Days Ahead to Predict", 1, 30, 7)

selected_df = stock_df[stock_df['Ticker'] == ticker]

st.subheader(f"Historical Stock Prices for {ticker}")
fig = px.line(selected_df, x='Date', y='Close', title=f"{ticker} Closing Prices")
st.plotly_chart(fig)

# ✅ Predict Future Prices
last_row = selected_df.iloc[-1]
pred_features = np.array([[last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume'],
                           last_row['MA_7'], last_row['MA_30'], last_row['Lag_1'], last_row['Lag_7']]])
pred_prices = [model.predict(pred_features)[0] for _ in range(days_ahead)]

dates_future = pd.date_range(start=selected_df['Date'].max(), periods=days_ahead+1)[1:]
pred_df = pd.DataFrame({'Date': dates_future, 'Predicted Price': pred_prices})

fig_pred = px.line(pred_df, x='Date', y='Predicted Price', title="Predicted Prices")
st.plotly_chart(fig_pred)

st.write("Predicted prices for the next days:")
st.dataframe(pred_df)
