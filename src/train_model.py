import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

def prepare_data_for_lstm(data_path='data/btcusd_with_indicators.csv', sequence_length=60):
    """
    Prepare data for LSTM model training
    
    Parameters:
    data_path (str): Path to the processed data CSV
    sequence_length (int): Number of time steps for LSTM
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Drop rows with NaN values (from indicator calculations)
    df = df.dropna()
    
    # Features to use
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 
                'MACD', 'Signal_Line', 'RSI', 'BB_Upper', 'BB_Lower', 'Volume_SMA']
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict closing price
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler

def build_lstm_model(input_shape):
    """
    Build LSTM model for price prediction
    
    Parameters:
    input_shape (tuple): Shape of input data
    
    Returns:
    tf.keras.Model: Compiled LSTM model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train, X_test, y_test, model_save_path='models/btcusd_lstm_model.h5'):
    """
    Train the LSTM model
    
    Parameters:
    X_train, X_test, y_train, y_test: Training and testing data
    model_save_path (str): Path to save the trained model
    """
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """
    Plot training history
    
    Parameters:
    history: Training history from model.fit()
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('models/training_history.png')
    plt.show()

if __name__ == "__main__":
    # Prepare data
    print("Preparing data for training...")
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm()
    
    # Train model
    print("Training LSTM model...")
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    print("Model training complete!")