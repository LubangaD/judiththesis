#!/usr/bin/env python3
"""
Model training script for cement forecasting application.
Trains ARIMA, MLP, and LSTM models on the provided data.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# Configuration
DATA_PATH = 'data/clean_data.csv'
LOOK_BACK = 12
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def load_and_prepare_data():
    """Load and prepare data for training."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def create_sequences(data, look_back, target_idx=0):
    """Create sequences for supervised learning."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, target_idx])
    return np.array(X), np.array(y)


def train_arima(data, target_col, order=(1, 1, 1)):
    """Train ARIMA model."""
    print("\n" + "=" * 60)
    print("Training ARIMA Model")
    print("=" * 60)
    
    # Prepare data
    y = data[target_col]
    
    # Split data
    train_size = int(len(y) * TRAIN_SPLIT)
    train, test = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    
    # Train model
    print(f"Fitting ARIMA{order}...")
    model = ARIMA(train, order=order)
    fitted_model = model.fit()
    
    print(fitted_model.summary())
    
    # Evaluate
    predictions = fitted_model.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mae = mean_absolute_error(test, predictions)
    
    print(f"\nTest RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    
    # Save model
    fitted_model.save('models/arima_cement.pkl')
    print("✓ Model saved: models/arima_cement.pkl")
    
    return fitted_model, rmse, mae


def build_mlp_model(input_shape):
    """Build MLP model architecture."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_mlp(X_train, y_train, X_test, y_test):
    """Train MLP model."""
    print("\n" + "=" * 60)
    print("Training MLP Model")
    print("=" * 60)
    
    # Flatten sequences for MLP
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Training shape: {X_train_flat.shape}")
    print(f"Test shape: {X_test_flat.shape}")
    
    # Build model
    model = build_mlp_model(X_train_flat.shape[1])
    print(model.summary())
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train_flat, y_train,
        validation_data=(X_test_flat, y_test),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    predictions = model.predict(X_test_flat, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\nTest RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    
    # Save model
    model.save('models/mlp_cement_model.keras')
    print("✓ Model saved: models/mlp_cement_model.keras")
    
    return model, history, rmse, mae


def build_lstm_model(input_shape):
    """Build LSTM model architecture."""
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, 
             input_shape=(input_shape[1], input_shape[2])),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_lstm(X_train, y_train, X_test, y_test):
    """Train LSTM model."""
    print("\n" + "=" * 60)
    print("Training LSTM Model")
    print("=" * 60)
    
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Build model
    model = build_lstm_model(X_train.shape)
    print(model.summary())
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    predictions = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\nTest RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    
    # Save model
    model.save('models/lstm_cement_model.keras')
    print("✓ Model saved: models/lstm_cement_model.keras")
    
    return model, history, rmse, mae


def main():
    """Main training function."""
    print("=" * 60)
    print("Cement Forecasting - Model Training")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Define target and features
    target_col = 'Cement'
    feature_cols = df.columns.tolist()
    target_idx = feature_cols.index(target_col)
    
    # Scale data
    print("\nScaling data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    print(f"Creating sequences (look_back={LOOK_BACK})...")
    X, y = create_sequences(scaled_data, LOOK_BACK, target_idx)
    
    print(f"Sequence shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    train_size = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train ARIMA
    arima_model, arima_rmse, arima_mae = train_arima(df, target_col)
    
    # Train MLP
    mlp_model, mlp_history, mlp_rmse, mlp_mae = train_mlp(X_train, y_train, X_test, y_test)
    
    # Train LSTM
    lstm_model, lstm_history, lstm_rmse, lstm_mae = train_lstm(X_train, y_train, X_test, y_test)
    
    # Save scaler and metadata
    print("\nSaving scaler and metadata...")
    joblib.dump(scaler, 'models/minmax_scaler.pkl')
    
    metadata = {
        'target': target_col,
        'columns': feature_cols,
        'look_back': LOOK_BACK,
        'model_performance': {
            'arima': {'rmse': arima_rmse, 'mae': arima_mae},
            'mlp': {'rmse': mlp_rmse, 'mae': mlp_mae},
            'lstm': {'rmse': lstm_rmse, 'mae': lstm_mae}
        }
    }
    joblib.dump(metadata, 'models/metadata.pkl')
    
    print("✓ Scaler saved: models/minmax_scaler.pkl")
    print("✓ Metadata saved: models/metadata.pkl")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"ARIMA  - RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}")
    print(f"MLP    - RMSE: {mlp_rmse:.2f}, MAE: {mlp_mae:.2f}")
    print(f"LSTM   - RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}")
    print("=" * 60)
    print("\nAll models trained successfully!")
    print("You can now run the Streamlit app: streamlit run app.py")


if __name__ == "__main__":
    main()
