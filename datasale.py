# -*- coding: utf-8 -*-
"""
Project: Mawsem - AI Seasonal Forecasting Engine
Description: Saudi Seasonal Sales Prediction using SARIMAX & Hijri Calendar
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ummalqura.hijri_date import HijriDate

# ==========================================
# 1. Data Ingestion & Setup
# ==========================================

# File path for the dataset
file_path = 'SuperMarket Analysis.csv'

if os.path.exists(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    # Convert 'Date' to standard datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✅ Success: '{file_path}' loaded correctly.")
else:
    print(f"❌ Error: '{file_path}' not found. Ensure it is in the same directory.")
    exit() # Stop execution if file is missing

# ==========================================
# 2. Localization & Hijri Engineering
# ==========================================

# English names for Hijri months
hijri_months_en = {
    1: 'Muharram', 2: 'Safar', 3: 'Rabi Al-Awwal', 4: 'Rabi Al-Thani',
    5: 'Jumada Al-Ula', 6: 'Jumada Al-Akhira', 7: 'Rajab', 8: 'Shaaban',
    9: 'Ramadan', 10: 'Shawwal', 11: 'Dhu Al-Qidah', 12: 'Dhu Al-Hijjah'
}

def get_hijri_data(date):
    """Converts Gregorian date to Hijri features."""
    hj = HijriDate(date.year, date.month, date.day, gr=True)
    month_name = hijri_months_en.get(hj.month)
    return hj.day, hj.month, month_name, hj.year

# Apply Hijri conversion
df['Hijri_Day'], df['Hijri_Month_Num'], df['Hijri_Month_Name'], df['Hijri_Year'] = zip(*df['Date'].map(get_hijri_data))

def assign_saudi_season(row):
    """Identifies Saudi religious and seasonal events."""
    if row['Hijri_Month_Num'] == 9:
        return 'Ramadan'
    elif row['Hijri_Month_Num'] == 10 and row['Hijri_Day'] <= 3:
        return 'Eid Al-Fitr'
    elif row['Hijri_Month_Num'] == 12 and row['Hijri_Day'] >= 1 and row['Hijri_Day'] <= 13:
        return 'Hajj Season'
    elif row['Date'].month in [12, 1, 2]:
        return 'Winter'
    else:
        return 'Regular'

# Apply Seasonality and simulate Saudi Cities
df['Season'] = df.apply(assign_saudi_season, axis=1)
df['City'] = np.random.choice(['Makkah', 'Riyadh', 'Jeddah', 'Dammam', 'Madinah'], size=len(df))

# ==========================================
# 3. Data Aggregation (Time-Series Prep)
# ==========================================

# Group by date to get daily totals
daily_sales = df.groupby('Date').agg({
    'Sales': 'sum',
    'Season': 'first',
    'City': 'first',
    'Hijri_Month_Num': 'first'
}).sort_values('Date')

# Set index and fill missing days (Resampling)
daily_sales = daily_sales.resample('D').asfreq()
daily_sales['Sales'] = daily_sales['Sales'].fillna(0)
daily_sales['Season'] = daily_sales['Season'].fillna('Regular')
daily_sales['City'] = daily_sales['City'].fillna('Makkah')

# ==========================================
# 4. AI Model Training (S-ARIMAX)
# ==========================================

# Prepare Exogenous variables (X)
daily_sales['Season_Code'] = daily_sales['Season'].astype('category').cat.codes
exog = daily_sales[['Season_Code']]

# Initialize SARIMAX (p,d,q) x (P,D,Q,s)
model = sm.tsa.statespace.SARIMAX(
    daily_sales['Sales'],
    exog=exog,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7) # Weekly seasonality
)

# Fit the model
print("🤖 AI Training in progress...")
results = model.fit(disp=False)

# Forecast the next 30 days
forecast_steps = 30
forecast_exog = pd.DataFrame({'Season_Code': [0] * forecast_steps},
                             index=pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=forecast_steps))
forecast = results.get_forecast(steps=forecast_steps, exog=forecast_exog)

print("✅ AI Training Complete. Predictions generated.")

# ==========================================
# 5. Visualization & Results
# ==========================================

forecast_series = forecast.predicted_mean
conf_int = forecast.conf_int()

plt.figure(figsize=(12, 6))

# Plot historical vs forecast
plt.plot(daily_sales['Sales'].tail(60), label='Actual Sales (History)', color='blue', linewidth=2)
plt.plot(forecast_series, label='AI Forecast (Next 30 Days)', color='red', linestyle='--', linewidth=2)

# Fill confidence interval
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)

plt.title('Mawsem: AI-Powered Seasonal Demand Forecast', fontsize=15)
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.legend()
plt.grid(True, alpha=0.3)

print("📊 Displaying results...")
plt.show()
