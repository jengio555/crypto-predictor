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
    data["macd"] = data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()
    data["macd_signal"] = data["macd"].ewm(span=9).mean()
    data["bollinger_upper"] = data["Close"].rolling(20).mean() + 2 * data["Close"].rolling(20).std()
    data["bollinger_lower"] = data["Close"].rolling(20).mean() - 2 * data["Close"].rolling(20).std()
    data["bollinger_width"] = data["bollinger_upper"] - data["bollinger_lower"]
    data["volume_ma"] = data["Volume"].rolling(10).mean()
    data["volume_ratio"] = data["Volume"] / data["volume_ma"]
    data["price_momentum"] = data["Close"].pct_change(5)
    data["volatility"] = data["returns"].rolling(10).std()
    data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["close_position"] = (data["Close"] - data["Low"]) / (data["High"] - data["Low"])
    return data.dropna()

def get_signal(crypto):
    ticker = yf.Ticker(crypto)
    train_data = ticker.history(period="730d", interval="1d")
    chart_data = ticker.history(period="1d", interval="1m")
    current_price = float(train_data["Close"].iloc[-1])

    train_data = calculate_features(train_data)

    feature_cols = [
        "ma_5", "ma_20", "ma_50", "ma_200",
        "rsi", "macd", "macd_signal",
        "bollinger_upper", "bollinger_lower", "bollinger_width",
        "Volume", "volume_ma", "volume_ratio",
        "price_momentum", "volatility",
        "high_low_range", "close_position"
    ]

    X = train_data[feature_cols].values.astype(float)
    y = train_data["returns"].values.astype(float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    ensemble = VotingRegressor([("rf", rf), ("gb", gb)])
    ensemble.fit(X_train, y_train)
    accuracy = ensemble.score(X_test, y_test)

    latest = scaler.transform([train_data[feature_cols].iloc[-1].values.astype(float)])
    prediction = ensemble.predict(latest)[0]
    signal = "LONG" if prediction > 0 else "SHORT"
    take_profit = current_price * 1.05
    stop_loss = current_price * 0.97

    rsi_val = float(train_data["rsi"].iloc[-1])
    if rsi_val > 70:
        rsi_note = "Overbought"
    elif rsi_val < 30:
        rsi_note = "Oversold"
    else:
        rsi_note = "Neutral"

    if abs(prediction) > 0.02:
        multiplier = "3X"
    elif abs(prediction) > 0.01:
        multiplier = "2X"
    else:
        multiplier = "1X"

    return signal, multiplier, current_price, take_profit, stop_loss, chart_data, rsi_val, rsi_note, accuracy

for left, right in pairs:
    col1, col2 = st.columns(2)
    for col, crypto in zip([col1, col2], [left, right]):
        with col:
            st.markdown("### " + crypto)
            try:
                signal, multiplier, current_price, take_profit, stop_loss, chart_data, rsi_val, rsi_note, accuracy = get_signal(crypto)
                color = "green" if signal == "LONG" else "red"
                st.markdown("**Signal:** :" + color + "[" + signal + "] | **Multiply:** " + multiplier)
                st.write("Entry: $" + str(round(current_price, 2)) + " | TP: $" + str(round(take_profit, 2)) + " | SL: $" + str(round(stop_loss, 2)))
                st.write("RSI: " + str(round(rsi_val, 2)) + " (" + rsi_note + ")")
                st.write("Model Accuracy: " + str(round(accuracy * 100, 2)) + "%")
                st.line_chart(chart_data["Close"])
            except Exception as e:
                st.write("Error: " + str(e))
            st.divider()
