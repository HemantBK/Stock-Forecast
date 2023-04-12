# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:14:34 2023

@author: shrey
"""

import pandas as pd
import streamlit as st
from pickle import load
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing 



data_close = load(open('df3.pkl','rb'))

from PIL import Image
image = Image.open("apple-stock.jpeg")
st.image(image)


html_temp="""
<div style ="background-color:Black;padding:10px">
<h2 style="color:white;text-align:center;"> Forecasting Apple Stocks
"""

st.markdown(html_temp,unsafe_allow_html=True) 

periods = st.number_input('Enter the number of Days to forecast: ',min_value=1, max_value=365)

datetime = pd.date_range('2019-12-30', periods=periods)
s = pd.Series(pd.date_range('2019-12-30', periods=periods))
date_df = pd.DataFrame(s.dt.date,columns=['Date'])


df2 = pd.read_csv('AAPL.csv')
Train = df2.head(1760)
Test = df2.tail(251)

hwe_model_mul_add = ExponentialSmoothing(Train["Close"],seasonal="mul",trend="add",seasonal_periods=251).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])


hwe_model_mul_add = ExponentialSmoothing(df2.Close,seasonal="add",trend="add",seasonal_periods=251).fit()

y_pred = hwe_model_mul_add.predict(start=len(Train), end=len(Train)+len(Test)-1)

forecast = hwe_model_mul_add.forecast(steps=periods)

st.title('Forecasted values for specified period')
forecast_df = pd.DataFrame(forecast)
forecast_df.columns = ['Close']

# Forecasted values
data_forecast = forecast_df.set_index(date_df.Date)
st.write(data_forecast)

#Average
st.subheader('Average price of stocks for specified time frame')
avg = pd.Series(forecast)
st.write(avg.mean())


# Assuming data_forecast contains the forecasted values of Apple stock
st.title('Visualizing Forecasted values for specified period')
fig, ax = plt.subplots(figsize=(15, 15))
ax.plot(data_forecast, color='Green')
ax.set_title('Apple Stock Forecast')
ax.set_xlabel('Date', color='Blue')
ax.set_ylabel('Stock Price', color='Blue')
ax.grid(True)
st.pyplot(fig)
plt.show()


st.title('Actual vs Predicted Values')
fig, ax = plt.subplots(figsize=(15, 15))
ax.plot(Test['Close'], color='Green')
ax.plot(y_pred, color='red')
ax.set_title('Apple Stock Forecast',fontsize=18)
ax.set_xlabel('Date', color='Blue')
ax.set_ylabel('Stock Price', color='Blue')
ax.grid(True)
st.pyplot(fig)
plt.show()