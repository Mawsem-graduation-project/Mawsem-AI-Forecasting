# 🌙 Mawsem: AI-Powered Seasonal Sales Forecasting

## 📌 Project Overview
**Mawsem** is a specialized forecasting engine designed for the Saudi retail market. The core objective of this AI component is to bridge the gap between Gregorian sales data and the **Saudi Hijri Calendar**, enabling businesses to predict demand surges during local seasons like **Ramadan, Eid Al-Fitr, and the Hajj season.**

## 🛠️ Technical Implementation
- **Calendar Integration:** Automated conversion from Gregorian to UmmAlQura Hijri calendar to capture the true lunar seasonality.
- **Feature Engineering:** Developed custom "Seasonality Logic" to categorize sales into local cultural events.
- **Machine Learning Model:** Implemented the **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with Exogenous factors) model. This allows the system to factor in external seasonal variables ($X$) alongside historical sales data.

## 📂 Repository Structure
- `notebooks/`: Contains the `.ipynb` file with the full data science workflow.
- `data/`: The dataset used for training and testing (`SuperMarket Analysis.csv`).
- `requirements.txt`: List of Python libraries required to run the engine.

## 📊 AI Output Analysis & Visualization
The forecasting engine generates a comprehensive **Time-Series Analysis** that visualizes the relationship between historical data and future trends.

### **Detailed Output Description:**
1. **Historical Trend Analysis (Blue Line):**
   This represents the actual recorded sales. It shows the daily fluctuations and helps identify the baseline performance of the retail store before applying AI adjustments.

2. **AI-Driven Forecast (Red Dashed Line):**
   This is the core predictive output. By analyzing historical peaks and the current Hijri season, the SARIMAX model projects the next 30 days of sales. This line demonstrates the "Seasonal Rhythm," showing expected spikes in demand during religious and national occasions.

3. **Risk & Uncertainty Mitigation (Shaded Area):**
   The model provides a 95% confidence interval, visually represented by a shaded boundary around the forecast. This provides decision-makers with a "Best-case" and "Worst-case" scenario, which is critical for inventory management and supply chain stability.

> **Note:** To view the live generated graph, please run the notebook provided in the `notebooks/` directory using Google Colab or Jupyter Lab.
