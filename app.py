
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Predictive Analytics",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Predictive Analytics Dashboard")
st.write("Forecast future sales using historical data and Machine Learning.")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Historical Sales Dataset",
    type=["csv"]
)

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("historical_data.csv")

# Validate columns
if not {"Date", "Sales"}.issubset(df.columns):
    st.error("CSV must contain Date and Sales columns.")
    st.stop()

# Data cleaning
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

df = df.dropna(subset=["Date", "Sales"])
df = df.sort_values("Date")
df = df.groupby("Date", as_index=False)["Sales"].sum()

if len(df) < 10:
    st.error("Please provide at least 10 valid historical records.")
    st.stop()

if not np.isfinite(df["Sales"]).all():
    st.error("Sales data contains invalid values.")
    st.stop()

if (df["Sales"] < 0).any():
    st.error("Sales values cannot be negative.")
    st.stop()

st.subheader("📋 Historical Sales Data")
st.dataframe(df, use_container_width=True)

# Dataset overview
st.subheader("📊 Sales Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Total Sales", f"{df['Sales'].sum():,.0f}")
col3.metric("Average Sales", f"{df['Sales'].mean():,.2f}")

# Prepare time-series features
# Assumes one observation per month
df["Month_Number"] = np.arange(len(df))

# Chronological train-test split
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[["Month_Number"]]
y_train = train["Sales"]

X_test = test[["Month_Number"]]
y_test = test["Sales"]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test predictions
test_predictions = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
r2 = r2_score(y_test, test_predictions)

st.subheader("🎯 Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric("R² Score", f"{r2:.3f}")

st.caption(
    "Evaluation uses the last 20% of records as test data. "
    "R² can be negative when predictions perform poorly."
)

# Actual vs predicted
st.subheader("📉 Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    test["Date"],
    y_test,
    marker="o",
    label="Actual Sales"
)

ax.plot(
    test["Date"],
    test_predictions,
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Actual vs Predicted Sales")
ax.legend()
ax.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Forecast future sales
st.subheader("🔮 Future Sales Forecast")

months = st.slider(
    "Select number of future months",
    min_value=1,
    max_value=12,
    value=6
)

# Retrain using all historical data
final_model = LinearRegression()

final_model.fit(
    df[["Month_Number"]],
    df["Sales"]
)

future_numbers = np.arange(
    len(df),
    len(df) + months
)

future_predictions = final_model.predict(
    pd.DataFrame({"Month_Number": future_numbers})
)

# Prevent negative sales predictions
future_predictions = np.maximum(future_predictions, 0)

last_date = df["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(1),
    periods=months,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": np.round(future_predictions, 2)
})

st.dataframe(forecast_df, use_container_width=True)

# Forecast chart
fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(
    df["Date"],
    df["Sales"],
    marker="o",
    label="Historical Sales"
)

ax2.plot(
    future_dates,
    future_predictions,
    marker="o",
    linestyle="--",
    label="Forecast Sales"
)

ax2.set_title("Future Sales Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Sales")
ax2.legend()
ax2.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig2)
plt.close(fig2)

# Download forecast
csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Forecast Report",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

st.success("Predictive Analytics completed successfully!")
