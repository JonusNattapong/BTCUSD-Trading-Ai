import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

def load_model_and_scaler(model_path='models/btcusd_lstm_model.h5', scaler_data_path='data/btcusd_with_indicators.csv'):
    """
    Load trained model and scaler
    
    Parameters:
    model_path (str): Path to saved model
    scaler_data_path (str): Path to data used for scaling
    
    Returns:
    tuple: (model, scaler, features)
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Recreate scaler
    df = pd.read_csv(scaler_data_path, index_col=0, parse_dates=True)
    df = df.dropna()
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 
                'MACD', 'Signal_Line', 'RSI', 'BB_Upper', 'BB_Lower', 'Volume_SMA']
    scaler = MinMaxScaler()
    scaler.fit_transform(df[features])
    
    return model, scaler, features

def predict_future_prices(model, scaler, features, last_sequence, days_to_predict=30):
    """
    Predict future prices using the trained model
    
    Parameters:
    model: Trained LSTM model
    scaler: Fitted MinMaxScaler
    features: List of feature names
    last_sequence: Last sequence of data for prediction
    days_to_predict (int): Number of days to predict
    
    Returns:
    np.array: Predicted prices
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        # Predict next price
        pred = model.predict(current_sequence.reshape(1, -1, len(features)), verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence (simplified - in practice you'd need to update all features)
        # For demo purposes, we'll just shift the sequence
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred[0]  # This is oversimplified
    
    return np.array(predictions)

def generate_trading_signals(predictions, current_price, threshold=0.02):
    """
    Generate trading signals based on predictions
    
    Parameters:
    predictions (np.array): Predicted prices
    current_price (float): Current BTC price
    threshold (float): Threshold for signal generation (2% default)
    
    Returns:
    str: Trading signal ('BUY', 'SELL', 'HOLD')
    """
    avg_prediction = np.mean(predictions)
    price_change = (avg_prediction - current_price) / current_price
    
    if price_change > threshold:
        return 'BUY'
    elif price_change < -threshold:
        return 'SELL'
    else:
        return 'HOLD'

def backtest_strategy(data_path='data/btcusd_with_indicators.csv', initial_balance=10000):
    """
    Backtest the trading strategy
    
    Parameters:
    data_path (str): Path to historical data
    initial_balance (float): Initial trading balance
    
    Returns:
    dict: Backtest results
    """
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = df.dropna()
    
    balance = initial_balance
    position = 0  # 0: no position, 1: long BTC
    trades = []
    
    for i in range(60, len(df)):  # Start after enough data for indicators
        current_price = df.iloc[i]['Close']
        
        # Simple strategy: Buy if RSI < 30, Sell if RSI > 70
        rsi = df.iloc[i]['RSI']
        
        if position == 0 and rsi < 30:
            # Buy
            position = balance / current_price
            balance = 0
            trades.append({'type': 'BUY', 'price': current_price, 'date': df.index[i]})
        elif position > 0 and rsi > 70:
            # Sell
            balance = position * current_price
            position = 0
            trades.append({'type': 'SELL', 'price': current_price, 'date': df.index[i]})
    
    # Close any remaining position
    if position > 0:
        final_price = df.iloc[-1]['Close']
        balance = position * final_price
        trades.append({'type': 'SELL', 'price': final_price, 'date': df.index[-1]})
    
    return {
        'final_balance': balance,
        'total_return': (balance - initial_balance) / initial_balance * 100,
        'trades': trades
    }

if __name__ == "__main__":
    # Load model and scaler
    try:
        model, scaler, features = load_model_and_scaler()
        print("Model loaded successfully!")
        
        # Load recent data for prediction
        df = pd.read_csv('data/btcusd_with_indicators.csv', index_col=0, parse_dates=True)
        df = df.dropna().tail(60)  # Last 60 days
        
        # Prepare last sequence
        scaled_data = scaler.transform(df[features])
        last_sequence = scaled_data
        
        # Predict future prices
        predictions = predict_future_prices(model, scaler, features, last_sequence, days_to_predict=7)
        
        # Inverse transform predictions
        pred_prices = scaler.inverse_transform(
            np.column_stack([predictions] + [np.zeros(len(predictions))] * (len(features)-1))
        )[:, 0]
        
        print(f"Predicted prices for next 7 days: {pred_prices}")
        
        # Generate trading signal
        current_price = df.iloc[-1]['Close']
        signal = generate_trading_signals(pred_prices, current_price)
        print(f"Current price: ${current_price:.2f}")
        print(f"Trading signal: {signal}")
        
    except FileNotFoundError:
        print("Model not found. Please train the model first using train_model.py")
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtest_strategy()
    print(f"Initial balance: $10000")
    print(f"Final balance: ${results['final_balance']:.2f}")
    print(f"Total return: {results['total_return']:.2f}%")
    print(f"Number of trades: {len(results['trades'])}")