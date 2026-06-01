from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from hijri_converter import Gregorian
import io

app = FastAPI(title="Mawsem AI Forecasting Service")


def meladi_to_hijri_month(date_str):
    """
    Convert Gregorian date string to Hijri (YYYY-MM) format based on Umm al-Qura calendar.
    """
    try:
        dt = pd.to_datetime(date_str)
        # Read the actual date directly since the Seeder is now accurate based on the Hijri calendar
        hijri_date = Gregorian(dt.year, dt.month, dt.day).to_hijri()
        return f"{hijri_date.year}-{hijri_date.month:02d}"
    except Exception:
        return None


@app.post("/forecast")
async def get_forecast(
        file: UploadFile = File(...),
        predictionTime: int = Form(30)  # Requested days or periods
):
    # 1. Read the transmitted CSV file from Laravel in memory
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

    # Validate required columns
    required_columns = {'sale_date', 'product_sku', 'quantity'}
    if not required_columns.issubset(df.columns):
        raise HTTPException(status_code=400, detail="CSV must contain sale_date, product_sku, and quantity")

    if df.empty:
        raise HTTPException(status_code=400, detail="Provided CSV file is empty.")

    # 2. Data Engineering for the Hijri Calendar
    df['hijri_month'] = df['sale_date'].apply(meladi_to_hijri_month)
    df = df.dropna(subset=['hijri_month'])

    # Aggregate sales monthly (Hijri) to capture Ramadan and Hajj seasons accurately
    monthly_series = df.groupby('hijri_month')['quantity'].sum().sort_index()

    # Validate sufficient data for seasonal SARIMA (Requires at least 12 Hijri months)
    if len(monthly_series) < 12:
        raise HTTPException(
            status_code=400,
            detail=f"Available Hijri monthly data ({len(monthly_series)} months) is insufficient for seasonal SARIMA. At least 12 Hijri months are required."
        )

    # 3. Build and train the SARIMA model
    # Seasonality m=12 because aggregation is now based on Hijri months.
    # Order (0,0,0) and Seasonal Order (1,0,0,12) are optimal configuration for pure seasonal data.
    try:
        model = SARIMAX(
            monthly_series.values,
            order=(0, 0, 0),
            seasonal_order=(1, 0, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)

        steps = 12
        forecast_values = model_fit.forecast(steps=steps)
        forecast_values = np.clip(forecast_values, 0, None)  # Prevent negative sales values

        # 4. Prepare future Hijri dates for the response
        last_hijri = monthly_series.index[-1]
        last_year, last_month = map(int, last_hijri.split('-'))

        forecast_output = []
        current_year = last_year
        current_month = last_month

        for val in forecast_values:
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

            forecast_output.append({
                "month_hijri": f"{current_year}-{current_month:02d}",
                "expected_sales": int(round(val))
            })

        return {
            "status": "success",
            "sku": str(df['product_sku'].iloc[0]),
            "months": forecast_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training or forecasting failed: {str(e)}")


@app.post("/dashboard")
async def get_dashboard_summary(file: UploadFile = File(...)):
    """
    Generate analytical insights and summary KPIs for the main dashboard
    based on the store's entire historical sales CSV data.
    """
    # 1. Read the transmitted CSV file from Laravel in memory
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

    # Validate required columns
    required_columns = {'sale_date', 'product_sku', 'quantity'}
    if not required_columns.issubset(df.columns):
        raise HTTPException(status_code=400, detail="CSV must contain sale_date, product_sku, and quantity")

    if df.empty:
        raise HTTPException(status_code=400, detail="Provided CSV file is empty.")

    try:
        # 1. Data Engineering & Transformation
        df['hijri_month'] = df['sale_date'].apply(meladi_to_hijri_month)
        df = df.dropna(subset=['hijri_month'])

        # 2. Calculate Main KPIs (Correct & Working)
        total_sales_quantity = int(df['quantity'].sum())

        monthly_grouped = df.groupby('hijri_month')['quantity'].sum()
        monthly_average = int(round(monthly_grouped.mean())) if not monthly_grouped.empty else 0
        peak_hijri_month = str(monthly_grouped.idxmax()) if not monthly_grouped.empty else "N/A"

        product_grouped = df.groupby('product_sku')['quantity'].sum()
        top_product_sku = str(product_grouped.idxmax()) if not product_grouped.empty else "N/A"
        top_product_qty = int(product_grouped.max()) if not product_grouped.empty else 0

        # 3. FIX: Calculate Seasonal Insights correctly
        # First, get the TOTAL sales for each unique Hijri month in history (e.g., "1445-09": 12000, "1446-09": 14000)
        total_per_hijri_month = df.groupby('hijri_month')['quantity'].sum().reset_index()

        # Extract the pure month number from the unique Hijri month string
        total_per_hijri_month['month_num'] = total_per_hijri_month['hijri_month'].apply(lambda x: int(x.split('-')[1]))

        # Now, average the totals across different years for each month number
        seasonal_averages = total_per_hijri_month.groupby('month_num')['quantity'].mean()

        ramadan_avg = int(round(seasonal_averages.get(9, 0)))
        hajj_avg = int(round(seasonal_averages.get(12, 0)))

        # Calculate average for regular months (excluding Ramadan 9 and Hajj 12)
        regular_months = seasonal_averages.drop([9, 12], errors='ignore')
        regular_avg = int(round(regular_months.mean())) if not regular_months.empty else 0

        return {
            "status": "success",
            "kpis": {
                "total_sales_quantity": total_sales_quantity,
                "monthly_average": monthly_average,
                "peak_hijri_month": peak_hijri_month,
                "top_performing_product": {
                    "sku": top_product_sku,
                    "total_quantity_sold": top_product_qty
                }
            },
            "seasonal_insights": {
                "ramadan_season_average": ramadan_avg,
                "hajj_season_average": hajj_avg,
                "regular_months_average": regular_avg
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
