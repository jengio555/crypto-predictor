import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("Crypto Prediction Dashboard")

cryptos = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "LINK-USD", "LTC-USD", "BCH-USD", "ZEN-USD"]

pairs = list(zip(cryptos[::2], cryptos[1::2]))

for left, right in pairs:
    col1, col2 = st.columns(2)
    for col, crypto in zip([col1, col2], [left, right]):
        with col:
            try:
                ticker = yf.Ticker(crypto)
                train_data = ticker.history(period="30d", interval="1d")
                chart_data = ticker.history(period="1d", interval="1m")
                current_price = float(train_data["Close"].iloc[-1])
                train_data["returns"] = train_data["Close"].pct_change().fillna
