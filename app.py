import streamlit as st
import yfinance as yf
import numpy as np
import requests
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from supabase import create_client

SUPABASE_URL = "https://uwhfboxiuvkorhhjeypy.supabase.co"
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(layout="wide")
st.title("Crypto Prediction")

import streamlit as st
import yfinance as yf
import numpy as np
import requests
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from supabase import create_client

SUPABASE_URL = "https://uwhfboxiuvkorhhjeypy.supabase.co"
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(layout="wide")
st.title("Crypto Prediction Dashboard")

total_balance = 18.97
min_allocation = 12.0

cryptos = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "LINK-USD", "LTC-USD"]
pairs = list(zip(cryptos[::2], cryptos[1::2]))


def send_telegram(msg):
    try:
        requests.post(
            "https://api.telegram.org/bot" + TELEGRAM_TOKEN + "/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        )
    except Exception:
        pass


def save_prediction(crypto, signal, prediction):
    try:
        supabase.table("predictions").insert({
            "crypto": crypto,
            "signal": signal,
            "prediction": float(prediction),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "was_correct": None
        }).execute()
    except Exception:
        pass


def get_historical_accuracy(crypto):
    try:
        result = supabase.table("predictions").select("*").eq("crypto", crypto).not_.is_("was_correct", "null").execute()
        if result.data and len(result.data) > 0:
            correct = sum(1 for r in result.data if r["was_correct"])
            return correct / len(result.data)
        return None
    except Exception:
        return None


def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=5)
        value = int(r.json()["data"][0]["value"])
        label = "Greed" if value > 50 else "Fear"
        return (value - 50) / 50, label
    except Exception:
        return 0.0, "Neutral"


@st.cache_data(ttl=3600)
def get_signal(crypto):
    ticker = yf.Ticker(crypto)
    data = ticker.history(period="30d", interval="1h")

    current_price = float(data["Close"].iloc[-1])

    data["returns"] = data["Close"].pct_change().fillna(0)
    data["ma_5"] = data["Close"].rolling(5).mean()
    data["ma_20"] = data["Close"].rolling(20).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = delta.clip(upper=0).abs().rolling(14).mean()
    data["rsi"] = 100 - (100 / (1 + gain / loss))

    data["macd"] = data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()
    data["volume_ma"] = data["Volume"].rolling(10).mean()
    data["momentum"] = data["Close"].pct_change(5)
    data["volatility"] = data["returns"].rolling(10).std()

    data = data.dropna()

    fear_greed, fg_label = get_fear_greed()

    features = ["ma_5", "ma_20", "rsi", "macd", "Volume", "volume_ma", "momentum", "volatility"]

    X = data[features].values.astype(float)
    y = data["returns"].values.astype(float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    latest = scaler.transform([data[features].iloc[-1].values.astype(float)])
    prediction = model.predict(latest)[0]

    combined = prediction + (fear_greed * 0.2)
    signal = "LONG" if combined > 0 else "SHORT"

    take_profit = current_price * 1.05
    stop_loss = current_price * 0.97

    rsi_val = float(data["rsi"].iloc[-1])
    ma5 = float(data["ma_5"].iloc[-1])
    ma20 = float(data["ma_20"].iloc[-1])

    if rsi_val > 70:
        rsi_note = "Overbought"
        close_alert = True
    elif rsi_val < 30:
        rsi_note = "Oversold"
        close_alert = True
    else:
        rsi_note = "Neutral"
        close_alert = False

    if signal == "LONG" and ma5 < ma20:
        close_alert = True
    elif signal == "SHORT" and ma5 > ma20:
        close_alert = True

    confidence = min(abs(combined) * 100, 100)

    if confidence > 66:
        confidence_label = "High Confidence"
        suggested_amount = max(total_balance * 0.20, min_allocation)
        leverage = 5
    elif confidence > 33:
        confidence_label = "Medium Confidence"
        suggested_amount = max(total_balance * 0.10, min_allocation)
        leverage = 3
    else:
        confidence_label = "Low Confidence"
        suggested_amount = max(total_balance * 0.05, min_allocation)
        leverage = 1

    can_afford = total_balance >= suggested_amount

    save_prediction(crypto, signal, prediction)
    historical_accuracy = get_historical_accuracy(crypto)

    return (signal, current_price, take_profit, stop_loss, rsi_val, rsi_note,
            accuracy, confidence, confidence_label, combined, suggested_amount,
            leverage, can_afford, close_alert, fg_label, historical_accuracy)


all_signals = []
for crypto in cryptos:
    try:
        result = get_signal(crypto)
        all_signals.append((crypto, result))
    except Exception:
        pass

all_signals.sort(key=lambda x: abs(x[1][9]), reverse=True)

affordable = [(c, r) for c, r in all_signals if r[12]]
top_3 = affordable[:3]

st.markdown("---")
st.subheader("Best Trades Right Now")

if top_3:
    for crypto, result in top_3:
        (signal, current_price, take_profit, stop_loss, rsi_val, rsi_note,
         accuracy, confidence, confidence_label, combined, suggested_amount,
         leverage, can_afford, close_alert, fg_label, historical_accuracy) = result

        color = "green" if signal == "LONG" else "red"
        status = "CLOSE NOW!" if close_alert else "HOLD"
        st.markdown("**" + crypto + "** - :" + color + "[" + signal + "] | $" +
                    str(round(suggested_amount, 2)) + " at " + str(leverage) + "x | " + status)
        if close_alert:
            send_telegram("CLOSE ALERT: " + crypto + " - Close your position now!")
else:
    st.warning("Balance too low for minimum allocation of $" + str(min_allocation))

st.markdown("---")
st.subheader("All Crypto Signals")

close_alerts = []
for left, right in pairs:
    col1, col2 = st.columns(2)
    for col, crypto in zip([col1, col2], [left, right]):
        with col:
            st.markdown("### " + crypto)
            try:
                (signal, current_price, take_profit, stop_loss, rsi_val, rsi_note,
                 accuracy, confidence, confidence_label, combined, suggested_amount,
                 leverage, can_afford, close_alert, fg_label, historical_accuracy) = get_signal(crypto)

                color = "green" if signal == "LONG" else "red"
                st.markdown("**Signal:** :" + color + "[" + signal + "] | " + confidence_label)
                st.write("Entry: $" + str(round(current_price, 2)) + " | TP: $" +
                         str(round(take_profit, 2)) + " | SL: $" + str(round(stop_loss, 2)))

                if can_afford:
                    st.write("Invest: $" + str(round(suggested_amount, 2)) + " at " + str(leverage) + "x leverage")
                else:
                    st.error("Need $" + str(round(suggested_amount, 2)) + " minimum")

                st.write("RSI: " + str(round(rsi_val, 2)) + " (" + rsi_note + ") | Market: " + fg_label)
                st.write("Model Accuracy: " + str(round(accuracy * 100, 2)) + "%")

                if historical_accuracy is not None:
                    st.write("Win Rate: " + str(round(historical_accuracy * 100, 2)) + "%")

                if close_alert:
                    st.warning("CLOSE NOW!")
                    close_alerts.append(crypto)

            except Exception as e:
                st.write("Error: " + str(e))
            st.divider()

if close_alerts:
    st.error("CLOSE ALERTS: " + ", ".join(close_alerts))
