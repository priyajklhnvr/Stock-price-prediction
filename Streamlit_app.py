import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import date

# Load trained model
model = load_model("Stock Predictions Model.keras")

st.header("📈 Stock Market Predictor")

# User input
stock = st.text_input("Enter Stock Symbol", "GOOG")

# Date range (up to yesterday)
start = "2015-01-01"
end = date.today()

# Download stock data
data = yf.download(stock, start=start, end=end)

st.subheader("Stock Data")
st.write(data)

# Train-test split (80%-20%)
data_train = pd.DataFrame(data.Close[: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80) :])

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)

past_100_days = data_train.tail(100)  # LSTM needs last 100 days memory
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

# ---------------- Moving Averages ---------------- #

st.subheader("Price vs MA50")
ma_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label="Price")
plt.plot(ma_50, label="MA50")
plt.legend()
st.pyplot(fig1)
plt.close()

st.subheader("Price vs MA50 vs MA100")
ma_100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label="Price")
plt.plot(ma_50, label="MA50")
plt.plot(ma_100, label="MA100")
plt.legend()
st.pyplot(fig2)
plt.close()

st.subheader("Price vs MA100 vs MA200")
ma_200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label="Price")
plt.plot(ma_100, label="MA100")
plt.plot(ma_200, label="MA200")
plt.legend()
st.pyplot(fig3)
plt.close()

# ---------------- Prepare Test Data ---------------- #

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100 : i])
    y.append(data_test_scale[i, 0])

x = np.array(x)
y = np.array(y)

# Prediction
predict = model.predict(x)

# Reverse scaling
scale = 1 / scaler.scale_[0]
predict = predict * scale
y = y * scale

# ---------------- Final Comparison Plot ---------------- #

st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, label="Original Price", color="green")
plt.plot(predict, label="Predicted Price", color="red")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)
plt.close()


final_df = pd.DataFrame(
    {"Original_Price": y.flatten(), "Predicted_Price": predict.flatten()}
)
# flatten() is used to convert the model output from a 2D array into a 1D array

st.subheader("Final Original vs Predicted Prices")
st.write(final_df.tail(15))


last_date = data.index[-1]

st.success(f"📅 Data available up to: {last_date.date()}")


# ---------------- Tomorrow Prediction ---------------- #

last_100_days = data.Close.tail(100).values
last_100_days = last_100_days.reshape(-1, 1)
last_100_days_scaled = scaler.transform(last_100_days)

X_tomorrow = np.array([last_100_days_scaled])

tomorrow_price = model.predict(X_tomorrow)

# Reverse scaling
tomorrow_price = tomorrow_price * scale

st.subheader("📌 Tomorrow's Predicted Price")
st.info(f"Predicted closing price for next trading day: ₹ {tomorrow_price[0][0]:.2f}")

st.subheader("Stock Summary")
st.write(f"Total records: {len(data)}")
st.write(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")


st.warning("⚠️ This project is for educational purposes only and not financial advice.")
