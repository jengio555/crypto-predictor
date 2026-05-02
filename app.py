import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("Crypto Prediction Dashboard")

cryptos = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "LINK-USD", "LTC-USD", "BCH-USD", "ZEN-USD"]

pairs = list(zip(cryptos[::2], cryptos[1::2]))

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_features(data):
    data["returns"] = data["Close"].pct_change().fillna(0)
    data["ma_5"] = data["Close"].rolling(5).mean()
    data["ma_20"] = data["Close"].rolling(20).mean()
    data["ma_50"] = data["Close"].rolling(50).mean()
    data["ma_200"] = data["Close"].rolling(200).mean()
    data["rsi"] = calculate_rsi(data["Close"])
    data["macd"] = data["Close"].ewm(span=12).mean() - data["Close
