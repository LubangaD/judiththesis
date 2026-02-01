"""
Reusable forecasting utilities for cement production prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union


def recursive_forecast(
    model, 
    last_window: np.ndarray, 
    steps: int, 
    target_idx: int, 
    is_lstm: bool = True
) -> np.ndarray:
    """
    Generate multi-step forecasts recursively using neural network models.
    
    This function uses the model's predictions as inputs for subsequent predictions,
    creating a recursive forecasting chain.
    
    Args:
        model: Trained Keras model (MLP or LSTM)
        last_window: Last known window of scaled data, shape (look_back, n_features)
        steps: Number of future steps to forecast
        target_idx: Index of target variable in feature array
        is_lstm: Boolean indicating if model is LSTM (3D input) or MLP (2D input)
    
    Returns:
        Array of predicted values (scaled), shape (steps,)
    
    Example:
        >>> preds = recursive_forecast(lstm_model, last_data, 12, 0, is_lstm=True)
        >>> print(preds.shape)  # (12,)
    """
    preds = []
    window = last_window.copy()
    
    for _ in range(steps):
        # Prepare input based on model type
        if is_lstm:
            # LSTM expects 3D input: (batch_size, timesteps, features)
            model_input = window[np.newaxis, :, :]
        else:
            # MLP expects 2D input: (batch_size, features)
            model_input = window.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(model_input, verbose=0)[0, 0]
        preds.append(prediction)
        
        # Update window: shift and insert new prediction
        window = np.roll(window, -1, axis=0)
        window[-1, target_idx] = prediction
    
    return np.array(preds)


def create_sequences(
    data: np.ndarray, 
    look_back: int, 
    target_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for supervised learning.
    
    Args:
        data: Time series data, shape (n_samples, n_features)
        look_back: Number of previous timesteps to use as input
        target_idx: Index of the target variable in the feature array
    
    Returns:
        X: Input sequences, shape (n_sequences, look_back, n_features)
        y: Target values, shape (n_sequences,)
    
    Example:
        >>> X, y = create_sequences(scaled_data, look_back=12, target_idx=0)
        >>> print(X.shape, y.shape)  # (n-12, 12, n_features), (n-12,)
    """
    X, y = [], []
    
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, target_idx])
    
    return np.array(X), np.array(y)


def generate_forecast_dates(
    last_date: pd.Timestamp, 
    steps: int, 
    freq: str = 'MS'
) -> pd.DatetimeIndex:
    """
    Generate future dates for forecasts.
    
    Args:
        last_date: Last known date in the historical data
        steps: Number of future periods to generate
        freq: Frequency string ('MS' for month start, 'D' for daily, etc.)
    
    Returns:
        DatetimeIndex of future dates
    
    Example:
        >>> dates = generate_forecast_dates(pd.Timestamp('2023-12-01'), 12)
        >>> print(dates[0])  # 2024-01-01
    """
    return pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]


def calculate_forecast_metrics(
    actual: np.ndarray, 
    predicted: np.ndarray
) -> dict:
    """
    Calculate common forecasting accuracy metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        Dictionary containing MAE, RMSE, MAPE, and R²
    
    Example:
        >>> metrics = calculate_forecast_metrics(y_true, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.2f}")
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def prepare_lstm_data(
    X: np.ndarray
) -> np.ndarray:
    """
    Ensure data is in correct shape for LSTM input.
    
    Args:
        X: Input data, shape (n_samples, look_back, n_features) or (n_samples, features)
    
    Returns:
        Reshaped data for LSTM, shape (n_samples, look_back, n_features)
    """
    if len(X.shape) == 2:
        # Assume flattened, need to determine look_back and n_features
        raise ValueError("Cannot automatically reshape 2D data. Please provide 3D data.")
    return X


def prepare_mlp_data(
    X: np.ndarray
) -> np.ndarray:
    """
    Flatten data for MLP input.
    
    Args:
        X: Input data, shape (n_samples, look_back, n_features)
    
    Returns:
        Flattened data for MLP, shape (n_samples, look_back * n_features)
    """
    if len(X.shape) == 3:
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)
    return X


def create_forecast_dataframe(
    dates: pd.DatetimeIndex, 
    predictions: np.ndarray, 
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    Create a formatted DataFrame for forecast results.
    
    Args:
        dates: Forecast dates
        predictions: Predicted values
        model_name: Name of the forecasting model
    
    Returns:
        DataFrame with dates as index and predictions as column
    
    Example:
        >>> df = create_forecast_dataframe(dates, preds, "ARIMA")
        >>> print(df.head())
    """
    return pd.DataFrame({
        f'{model_name}_Forecast': predictions
    }, index=dates)


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler,
    target_idx: int,
    n_features: int
) -> np.ndarray:
    """
    Inverse transform scaled predictions back to original scale.
    
    Args:
        predictions: Scaled predictions
        scaler: Fitted scaler object (e.g., MinMaxScaler)
        target_idx: Index of target variable
        n_features: Total number of features
    
    Returns:
        Predictions in original scale
    """
    # Create dummy array with same shape as original features
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, target_idx] = predictions
    
    # Inverse transform
    inversed = scaler.inverse_transform(dummy)
    
    return inversed[:, target_idx]
