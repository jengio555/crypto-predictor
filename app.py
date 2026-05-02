import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title('Crypto Prediction Dashboard')
st.write('Live predictions updated every minute')

cryptos = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD', 'LINK-USD', 'LTC-USD', 'BCH-USD', 'ZEN-USD']

for crypto in cryptos:
    try:
        ticker = yf.Ticker(crypto)
        data = ticker.history(period='30d', interval='1d')
        current_price = float(data['Close'].iloc[-1])
        data['returns'] = data['Close'].pct_change().fillna(0)
        data['ma_5'] = data['Close'].rolling(5).mean()
        data = data.dropna()
        
        X = np.column_stack([data['ma_5'].values.astype(float), data['Volume'].values.astype(float)])
        y = data['returns'].values.astype(float)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        prediction = model.predict([[float(data['ma_5'].iloc[-1]), float(data['Volume'].iloc[-1])]])[0]
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
        st.write(f'Entry: ${current_price:.2f} | Take Profit: ${take_profit:.2f} | Stop Loss: ${stop_loss:.2f}')
        st.divider()
    except Exception as e:
        st.write(f'{crypto}: Error - {str(e)}')
