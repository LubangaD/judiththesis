#!/usr/bin/env python3
"""
Setup script for cement forecasting application.
Creates sample data and metadata if they don't exist.
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def create_sample_data():
    """Create sample historical data for demonstration."""
    print("Creating sample data...")
    
    # Generate sample dates (monthly from 2010 to 2023)
    dates = pd.date_range(start='2010-01-01', end='2023-12-01', freq='MS')
    n = len(dates)
    
    # Generate synthetic data with trend, seasonality, and noise
    np.random.seed(42)
    
    # Cement production (main target)
    trend = np.linspace(1000, 2200, n)
    seasonality = 150 * np.sin(np.arange(n) * 2 * np.pi / 12)
    noise = np.random.normal(0, 50, n)
    cement = trend + seasonality + noise
    
    # GDP (correlated with cement)
    gdp_base = 450000 + (cement - 1000) * 350
    gdp = gdp_base + np.random.normal(0, 15000, n)
    
    # Inflation rate
    inflation = 2.5 + 0.5 * np.sin(np.arange(n) * 2 * np.pi / 24) + np.random.normal(0, 0.3, n)
    
    # Construction activity (highly correlated with cement)
    construction = 500 + (cement - 1000) * 0.4 + np.random.normal(0, 30, n)
    
    # Imports
    imports = 200 + (cement - 1000) * 0.15 + np.random.normal(0, 20, n)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Cement': cement,
        'GDP': gdp,
        'Inflation': inflation,
        'Construction': construction,
        'Imports': imports
    }, index=dates)
    
    # Ensure positive values
    df = df.clip(lower=0)
    
    # Save to CSV
    data_path = Path('data/clean_data.csv')
    data_path.parent.mkdir(exist_ok=True)
    df.to_csv(data_path)
    
    print(f"✓ Sample data created: {data_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def create_sample_metadata():
    """Create sample metadata file."""
    print("\nCreating sample metadata...")
    
    metadata = {
        'target': 'Cement',
        'columns': ['Cement', 'GDP', 'Inflation', 'Construction', 'Imports'],
        'look_back': 12,
        'model_info': {
            'arima_order': (1, 1, 1),
            'mlp_architecture': [128, 64, 32],
            'lstm_units': 50,
        },
        'data_info': {
            'train_start': '2010-01-01',
            'train_end': '2021-12-01',
            'test_start': '2022-01-01',
            'test_end': '2023-12-01',
            'frequency': 'MS',
            'n_features': 5,
        }
    }
    
    # Save metadata
    models_path = Path('models')
    models_path.mkdir(exist_ok=True)
    
    metadata_path = models_path / 'metadata.pkl'
    joblib.dump(metadata, metadata_path)
    
    print(f"✓ Metadata created: {metadata_path}")
    print(f"  Target: {metadata['target']}")
    print(f"  Features: {len(metadata['columns'])}")
    print(f"  Look-back: {metadata['look_back']} months")
    
    return metadata


def check_requirements():
    """Check if required packages are installed."""
    print("\nChecking requirements...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'tensorflow',
        'statsmodels',
        'shap',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed")
        return True


def main():
    """Main setup function."""
    print("=" * 60)
    print("Cement Forecasting Application - Setup")
    print("=" * 60)
    
    # Check if data exists
    data_path = Path('data/clean_data.csv')
    if data_path.exists():
        print(f"\n✓ Data file already exists: {data_path}")
        overwrite = input("Overwrite with sample data? (y/N): ").lower()
        if overwrite == 'y':
            create_sample_data()
    else:
        create_sample_data()
    
    # Check if metadata exists
    metadata_path = Path('models/metadata.pkl')
    if metadata_path.exists():
        print(f"\n✓ Metadata file already exists: {metadata_path}")
        overwrite = input("Overwrite with sample metadata? (y/N): ").lower()
        if overwrite == 'y':
            create_sample_metadata()
    else:
        create_sample_metadata()
    
    # Check requirements
    check_requirements()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train your models (see train_models.py)")
    print("2. Run the application: streamlit run app.py")
    print("\nNote: You need to train models before running the app.")
    print("The sample data is for demonstration purposes only.")
    print("=" * 60)


if __name__ == "__main__":
    main()
