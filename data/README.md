# Data Directory

This directory contains the historical data used for forecasting.

## Expected File

**clean_data.csv**: Historical manufacturing activity data

## Data Format

The CSV file should have:
- **Date column** as the index (datetime format)
- **Target variable** (e.g., Cement production)
- **Additional features** used in the models

### Example Structure

```csv
Date,Cement,GDP,Inflation,Construction,Imports
2010-01-01,1250.5,450000,2.3,580.2,230.1
2010-02-01,1280.3,452000,2.5,590.5,235.4
2010-03-01,1310.7,455000,2.4,600.8,240.2
...
2023-12-01,2150.4,780000,3.1,950.3,420.5
```

## Requirements

1. **Date Index**: 
   - Must be parseable as datetime
   - Should be in ascending order
   - Preferably monthly frequency (Month Start)

2. **Target Variable**:
   - Numeric values (float or int)
   - No missing values (or handle appropriately)
   - Column name must match `metadata.pkl`

3. **Additional Features**:
   - All features used during model training
   - Same order as in `metadata.pkl`
   - Numeric values only

## Data Preprocessing

If you need to preprocess your data:

```python
import pandas as pd

# Load raw data
df = pd.read_csv('raw_data.csv')

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sort by date
df.sort_index(inplace=True)

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Save clean data
df.to_csv('data/clean_data.csv')
```

## Sample Data

If you don't have data yet, you can create sample data:

```python
import pandas as pd
import numpy as np

# Generate sample dates
dates = pd.date_range(start='2010-01-01', end='2023-12-01', freq='MS')

# Generate sample data with trend and seasonality
n = len(dates)
trend = np.linspace(1000, 2000, n)
seasonality = 100 * np.sin(np.arange(n) * 2 * np.pi / 12)
noise = np.random.normal(0, 50, n)

cement = trend + seasonality + noise
gdp = trend * 400 + np.random.normal(0, 10000, n)
inflation = 2.5 + np.random.normal(0, 0.5, n)

# Create DataFrame
df = pd.DataFrame({
    'Cement': cement,
    'GDP': gdp,
    'Inflation': inflation
}, index=dates)

# Save
df.to_csv('data/clean_data.csv')
```

## Notes

- Ensure your data covers a sufficient time period for training
- At least 2-3 years of monthly data is recommended
- More features may improve model performance but increase complexity
- Keep feature engineering consistent between training and deployment
