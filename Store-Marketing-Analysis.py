import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Read East West sheet from excel
store_data_ew = pd.read_excel("D:\Top Mentor\Assignments\Client data (Social Media Co.).xlsx", sheet_name='East_West', date_format='Dt_Customer')
print(store_data_ew.head(10))

# Clean the data
store_data_ew['Dt_Customer'] = pd.to_datetime(store_data_ew['Dt_Customer'], errors='coerce').dt.strftime('%Y-%m-%d')
store_data_ew[' Income '] = store_data_ew[' Income '].replace('[\$,]', '', regex=True).astype(float)
store_data_ew['Region'] = store_data_ew['Region'].replace({'eas': 'East', 'wst': 'West', 'Eas': 'East', 'Wst': 'West', 'east': 'East', 'west': 'West', 'WEST':'West'})
print(store_data_ew['Region'].unique())
store_data_ew.columns = store_data_ew.columns.str.strip()
print(store_data_ew.columns)

# Read North South sheet from excel
store_data_ns = pd.read_excel("D:\Top Mentor\Assignments\Client data (Social Media Co.).xlsx", sheet_name='North_South', date_format='Dt_Customer')

# Clean the data
store_data_ns['Dt_Customer'] = pd.to_datetime(store_data_ns['Dt_Customer'], errors='coerce').dt.strftime('%Y-%m-%d')
store_data_ns[' Income '] = store_data_ns[' Income '].replace('[\$,]', '', regex=True).astype(float)
store_data_ns['Region'] = store_data_ns['Region'].str.lower().replace({'nor': 'NORTH','north':'NORTH','soth': 'SOUTH','south':'SOUTH'})
print(store_data_ew['Region'].unique())
store_data_ns.columns = store_data_ns.columns.str.strip()
print(store_data_ns.columns)

# Reset the index of both data frame for memory management
store_data_ew = store_data_ew.reset_index(drop=True)  # Reset index and drop the old index
store_data_ns = store_data_ns.reset_index(drop=True)  # Reset index and drop the old index

# Join both data frames
final_store_data = pd.concat([store_data_ew,store_data_ns],axis = 0)
print(final_store_data.columns)
print(final_store_data.shape)
print(final_store_data.head(30))

print(final_store_data['Region'].unique())

# Read Stores data
df = pd.read_csv("D:\Top Mentor\Assignments\Stores Data.csv")
print(df.head(2))
print(df.shape)

final_store_data.to_csv("final_store_data.csv")
df.to_csv("df.csv")

# Read CSV files
final_store_data = pd.read_csv('final_store_data.csv', memory_map=True)
df = pd.read_csv('df.csv', memory_map=True)

# Join data frames using store code
joined_data = pd.merge(final_store_data, df, left_on='Store_Code', right_on='STORECODE')

print(joined_data.shape)
print(joined_data.head(5))

print(joined_data.columns)
print(joined_data['Store_Code'].unique())

data_for_forecast = joined_data.groupby(['Dt_Customer','STORECODE'])['Income'].mean().reset_index().sort_values('Dt_Customer',ascending  = True)
print(data_for_forecast.head(30))


plt.figure(figsize=(12,6))
sns.histplot(data_for_forecast['Income'], kde=True)
plt.title("Distribution of monthly income")
plt.show()


data = pd.read_csv('D:\Top Mentor\Assignments\data_for_forecast.csv',date_format='Dt_Customer')
print(data.head(5))
print(data.dtypes)

# Drop unnamed column
data.drop('Unnamed: 0', axis =1, inplace = True)
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])
print(data.dtypes)

data = data.sort_values(['STORECODE','Dt_Customer'], ascending  = [True, True])
print(data.head(5))

forecasts = {}
for store, group in data.groupby('STORECODE'):
    # Prepare time series data
    ts = group.set_index('Dt_Customer')['Income']

    # Train ARIMA model
    model = ARIMA(ts, order=(3, 1, 2))  # Adjust order as needed
    model_fit = model.fit()

    # Forecast next 15 days
    forecast = model_fit.forecast(steps=15)

    # Save results
    forecasts[store] = forecast

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(ts, label="Historical")
    plt.plot(pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=15), forecast, label="Forecast", color='red')
    plt.title(f"Income Forecast for Store {store}")
    plt.legend()
    plt.show()

    # Combine forecasts for all stores
    forecast_df = pd.DataFrame({
        store: forecast for store, forecast in forecasts.items()
    }, index=pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=1))

    print(forecast_df)
