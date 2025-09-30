# Ex.No: 6               HOLT WINTERS METHOD
### Date: 30/09/2025



### AIM:
To implement the Holt-Winters Exponential Smoothing method for forecasting sales data and evaluate the model’s performance.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load AirPassengers dataset via R datasets
data = sm.datasets.get_rdataset("AirPassengers", "datasets").data

# Convert to datetime index
data['time'] = pd.date_range(start='1949-01', periods=len(data), freq='M')
data = data.set_index('time')
data = data.rename(columns={"value": "Passengers"})

# Plot
data.plot(title="Monthly Air Passengers")

# Seasonal Decomposition
seasonal_decompose(data, model="multiplicative").plot()
plt.show()

# Train-Test Split
train = data[:'1958-12-01']
test = data['1959-01-01':]

# Holt-Winters model
hwmodel = ExponentialSmoothing(
    train["Passengers"],
    trend="add",
    seasonal="mul",
    seasonal_periods=12
).fit()

# Forecast for test length
test_pred = hwmodel.forecast(len(test))

# Plot
train["Passengers"].plot(label="Train", legend=True, figsize=(10,6))
test["Passengers"].plot(label="Test", legend=True)
test_pred.plot(label="Predicted", legend=True)

# RMSE
rmse = np.sqrt(mean_squared_error(test, test_pred))
print("RMSE:", rmse)



```
### OUTPUT:

### DATASET
<BR>
<BR>
<img width="552" height="455" alt="image" src="https://github.com/user-attachments/assets/5bfe0fa9-a561-48dc-8b3f-b467a3119989" />


<BR><BR>



### SEASONAL DECOMPOSITION:

<BR>
<BR>
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/ea8d1299-4168-42c9-8003-87975f069e80" />

<BR>
<BR>

### TEST_PREDICTION
<BR>
<img width="451" height="102" alt="image" src="https://github.com/user-attachments/assets/59a432c0-7276-4c5f-9a98-8a20cb059e8e" />


<BR>



### FINAL_PREDICTION

<BR>
<img width="831" height="525" alt="image" src="https://github.com/user-attachments/assets/01d48711-be10-40a5-9240-b469abacaeca" />


<BR>

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
