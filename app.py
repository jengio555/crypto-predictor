import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title('Crypto Prediction Dashboard')

cryptos = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD', 'LINK-USD', 'LTC-USD', 'BCH-USD', 'ZEN-USD']

pairs = list(zip(cryptos[::2], cryptos[1::2]))

for left, right in pairs:
    col1, col2 = st.columns(2)
    for col, crypto in zip([col1, col2], [left, right]):
        with col:
            try:
                ticker = yf.Ticker(crypto)
                train_data = ticker.history(period='30d', interval='1d')
                chart_data = ticker.history(period='1d', interval='1m')
                current_price = float(train_data['Close'].iloc[-1])
                train_data['returns'] = train_data['Close'].pct_change().fillna(0)
                train_data['ma_5'] = train_data['Close'].rolling(5).mean()
                train_data = train_data.dropna()
                X = np.column_stack([train_data['ma_5'].values.astype(float), train_data['Volume'].values.astype(float)])
                y = train_data['returns'].values.astype(float)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                prediction = model.predict([[float(train_data['ma_5'].iloc[-1]), float(train_data['Volume'].iloc[-1])]])[0]
                signal = 'LONG' if prediction > 0 else 'SHORT'
                take_profit = current_price * 1.05
                stop_loss = current_price * 0.97
                if abs(prediction) > 0.02:
                    multiplier = '3X'
                elif abs(prediction) > 0.01:
                    multiplier = '2X'
                else:
                    multiplier = '1X'
                color = 'green' if signal == 'LONG' else 'red'
                st.markdown(f'### {crypto}')
                st.markdown(f'**Signal:** :{color}[{signal}] | **Multiply:** {multiplier}')
                st.write(f'Entry: ${current_price:.2f} | TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}')
                st
