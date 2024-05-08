import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Set the start date to obtain historical data

start_date = datetime.now() - timedelta(days=150)  # Get data for the past year
start_date = start_date.strftime('%Y-%m-%d')

# End date is today
end_date = datetime.now().strftime('%Y-%m-%d')

# Download historical stock data
stock = 'MARA'
data = yf.download(stock, start=start_date, end=end_date)

# Drop missing values
data.dropna(inplace=True)

# Prepare training data
date_train = pd.DataFrame(data['Close'])
scaler = MinMaxScaler(feature_range=(0, 1))
date_train_scale = scaler.fit_transform(date_train)

x_train, y_train = [], []
for i in range(100, date_train_scale.shape[0]):
    x_train.append(date_train_scale[i-100:i])
    y_train.append(date_train_scale[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=60, batch_size=32, verbose=1)

# Predict today's closing price
last_100_days = date_train.tail(100)
data_today = last_100_days.values.reshape(-1, 1)
data_today = scaler.transform(data_today)
x_today = np.array([data_today])
x_today = np.reshape(x_today, (x_today.shape[0], x_today.shape[1], 1))

predicted_price = model.predict(x_today)
predicted_price = scaler.inverse_transform(predicted_price)

original_price = data['Close'].iloc[-1]

print("Original price: ${:.2f}".format(original_price))
print("Predicted price: ${:.2f}".format(predicted_price[0][0]))