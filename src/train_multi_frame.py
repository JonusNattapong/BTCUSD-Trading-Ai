import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import yfinance as yf
import os

def download_multi_timeframe_data():
    """
    Download BTCUSD data for multiple timeframes
    """
    timeframes = {
        '1h': ('1h', 720),  # 30 days of hourly data
        '4h': ('4h', 180),  # 30 days of 4-hour data
        '1d': ('1d', 365)   # 1 year of daily data
    }

    data_frames = {}

    for tf_name, (interval, period) in timeframes.items():
        print(f"Downloading {tf_name} timeframe data...")

        # Try different tickers
        tickers = ['BTC-USD', 'BTCUSD=X', 'BTC-USD.CC']
        data = None

        for ticker in tickers:
            try:
                print(f"Trying ticker: {ticker}")
                btc = yf.Ticker(ticker)
                data = btc.history(period=f"{period}d", interval=interval)
                if not data.empty:
                    print(f"Successfully downloaded {tf_name} data using ticker: {ticker}")
                    break
            except Exception as e:
                print(f"Failed with ticker {ticker}: {e}")
                continue

        if data is None or data.empty:
            print(f"Failed to download {tf_name} data. Creating sample data.")
            # Create sample data for demonstration
            dates = pd.date_range(end=pd.Timestamp.now(), periods=period, freq=tf_name if tf_name != '4h' else '4H')
            np.random.seed(42)
            base_price = 40000
            prices = []
            current_price = base_price

            for _ in dates:
                change = np.random.normal(0, 0.02)
                current_price *= (1 + change)
                prices.append(current_price)

            data = pd.DataFrame({
                'Open': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'Close': prices,
                'Adj Close': prices,
                'Volume': [np.random.randint(1000000, 10000000) for _ in prices]
            }, index=dates)

        data_frames[tf_name] = data

    return data_frames

def add_technical_indicators(df):
    """
    Add technical indicators to dataframe
    """
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()

    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

    return df

def prepare_multi_timeframe_data(sequence_lengths={'1h': 24, '4h': 12, '1d': 30}):
    """
    Prepare data for multi-timeframe LSTM model

    Parameters:
    sequence_lengths (dict): Sequence lengths for each timeframe

    Returns:
    tuple: (X_train_dict, X_test_dict, y_train, y_test, scalers)
    """
    # Download data for all timeframes
    data_frames = download_multi_timeframe_data()

    processed_data = {}
    scalers = {}

    # Process each timeframe
    for tf_name, df in data_frames.items():
        print(f"Processing {tf_name} timeframe...")

        # Add technical indicators
        df_with_indicators = add_technical_indicators(df)

        # Drop NaN values
        df_clean = df_with_indicators.dropna()

        # Features to use
        features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                    'MACD', 'Signal_Line', 'RSI', 'BB_Upper', 'BB_Lower', 'Volume_SMA']

        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_clean[features])

        processed_data[tf_name] = {
            'data': scaled_data,
            'df': df_clean
        }
        scalers[tf_name] = scaler

    # Find the minimum length across all timeframes for alignment
    min_length = min(len(data['data']) for data in processed_data.values())
    seq_len_1h = sequence_lengths['1h']
    seq_len_4h = sequence_lengths['4h']
    seq_len_1d = sequence_lengths['1d']

    # Prepare sequences for each timeframe
    X_sequences = {}
    for tf_name, data_dict in processed_data.items():
        scaled_data = data_dict['data']
        seq_len = sequence_lengths[tf_name]

        X, y = [], []
        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i-seq_len:i])
            y.append(scaled_data[i, 0])  # Predict closing price

        X_sequences[tf_name] = np.array(X)
        if tf_name == '1d':  # Use daily as target
            y_target = np.array(y)

    # Align sequences (use the last part where all timeframes have data)
    min_sequences = min(len(seq) for seq in X_sequences.values())

    X_train_dict = {}
    X_test_dict = {}

    for tf_name, sequences in X_sequences.items():
        X_train_dict[tf_name] = sequences[:int(0.8 * min_sequences)]
        X_test_dict[tf_name] = sequences[int(0.8 * min_sequences):min_sequences]

    y_train = y_target[:int(0.8 * min_sequences)]
    y_test = y_target[int(0.8 * min_sequences):min_sequences]

    return X_train_dict, X_test_dict, y_train, y_test, scalers

def build_multi_timeframe_model(input_shapes):
    """
    Build multi-timeframe LSTM model

    Parameters:
    input_shapes (dict): Input shapes for each timeframe

    Returns:
    tf.keras.Model: Multi-timeframe model
    """
    # Input layers for each timeframe
    inputs = {}
    lstm_outputs = []

    for tf_name, shape in input_shapes.items():
        inputs[tf_name] = Input(shape=shape, name=f'{tf_name}_input')

        # LSTM layers for each timeframe
        x = LSTM(32, return_sequences=True)(inputs[tf_name])
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.2)(x)

        lstm_outputs.append(x)

    # Concatenate outputs from all timeframes
    concatenated = Concatenate()(lstm_outputs)

    # Final dense layers
    x = Dense(64, activation='relu')(concatenated)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1)(x)

    # Create model
    model = Model(inputs=list(inputs.values()), outputs=output)

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

def train_multi_timeframe_model(X_train_dict, X_test_dict, y_train, y_test, model_save_path='models/btcusd_multi_frame_model.h5'):
    """
    Train the multi-timeframe model
    """
    # Get input shapes
    input_shapes = {tf_name: X_train.shape[1:] for tf_name, X_train in X_train_dict.items()}

    # Build model
    model = build_multi_timeframe_model(input_shapes)
    print("Multi-timeframe model summary:")
    model.summary()

    # Prepare training data (separate inputs for each timeframe)
    train_inputs = [X_train_dict[tf_name] for tf_name in ['1h', '4h', '1d']]
    test_inputs = [X_test_dict[tf_name] for tf_name in ['1h', '4h', '1d']]

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

    # Train model
    history = model.fit(
        train_inputs, y_train,
        validation_data=(test_inputs, y_test),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    return model, history

def plot_training_history(history, save_path='models/multi_frame_training_history.png'):
    """
    Plot training history
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Multi-Timeframe Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Training Multi-Timeframe BTCUSD Trading AI...")

    # Prepare data
    print("Preparing multi-timeframe data...")
    X_train_dict, X_test_dict, y_train, y_test, scalers = prepare_multi_timeframe_data()

    print(f"Training sequences - 1h: {X_train_dict['1h'].shape}, 4h: {X_train_dict['4h'].shape}, 1d: {X_train_dict['1d'].shape}")

    # Train model
    print("Training multi-timeframe model...")
    model, history = train_multi_timeframe_model(X_train_dict, X_test_dict, y_train, y_test)

    # Plot training history
    plot_training_history(history)

    print("Multi-timeframe model training complete!")

    # Save scalers for later use
    import pickle
    with open('models/multi_frame_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    print("Scalers saved to models/multi_frame_scalers.pkl")