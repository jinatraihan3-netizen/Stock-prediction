# Stock-prediction
Title: Stock Price Forecasting for Multiple Tickers Interface: Built with Streamlit Functionality:
Lets users select up to 5 stock tickers (e.g., AAPL, MSFT).
Choose a forecasting model from ARIMA, SARIMA, Prophet, or LSTM.
Define a date range for historical data collection.
Visualize historical stock prices and forecasted trends.
📈 Forecasting Models Used ARIMA (AutoRegressive Integrated Moving Average)
Used for univariate time series forecasting.
Assumes linearity and requires stationary data.
order=(5,1,0) indicates the AR(5), I(1), MA(0) components.
Forecasts for 365 days and plots with historical prices.
SARIMA (Seasonal ARIMA)
Extends ARIMA to include seasonality.
order=(1,1,1) and seasonal_order=(1,1,1,12) model yearly seasonality.
Good for time series with seasonal patterns.
Prophet (by Facebook/Meta)
A robust model for time series forecasting with daily, weekly, and yearly seasonality.
Requires data with columns named 'ds' (date) and 'y' (value).
Produces forecasts with uncertainty intervals.
LSTM (Long Short-Term Memory Neural Network)
Deep learning model suitable for sequential data.
Normalizes the data using MinMaxScaler.
Trains an LSTM model with 2 layers and 1 Dense output layer.
Uses a 60-time-step sequence to predict the next value.
