import numpy as np 
import pandas as pd 
from fbprophet import Prophet

#%matplotlib inline

df = pd.read_csv("./avocado.csv")

df.head()

df.groupby('type').groups

PREDICTION_TYPE = 'conventional'
df = df[df.type == PREDICTION_TYPE]

df['Date'] = pd.to_datetime(df['Date'])

regions = df.groupby(df.region)

print("Total regions :", len(regions))
print("-------------")

for name, group in regions:
    print(name, " : ", len(group))
    
PREDICTING_FOR = "TotalUS"

date_price = regions.get_group(PREDICTING_FOR)[['Date', 'AveragePrice']].reset_index(drop=True)
date_price.plot(x='Date', y='AveragePrice', kind="line")
date_price = date_price.rename(columns={'Date':'ds', 'AveragePrice':'y'})

m = Prophet()
m.fit(date_price)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-30:]
res = data.to_json(orient='records', date_format='iso')

fig1 = m.plot(forecast)

import ipdb;ipdb.set_trace()
