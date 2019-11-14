import numpy as np 
import pandas as pd
from fbprophet import Prophet
from flask import Flask
import requests
from flask_restplus import Api, Resource, fields
import json

server = Flask(__name__)
api = Api(server, version='1.0', title='Pretty Cool API!', description='Awesome API')

#Web Service
avocado = api.namespace('avocado', description='Avocado Price Prediction')
avocado_model = api.model('Movie', {
    'ds': fields.Date(readOnly=True),
    'yhat': fields.Float(required=True),
    'yhat_lower': fields.Float(required=True),
    'yhat_upper': fields.Float(required=True)
})
@avocado.route('/')
class Avocado(Resource):
    @avocado.doc('predict_price')
    @avocado.marshal_list_with(avocado_model)
    def get(self):
        
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
        res_json = json.loads(res)
        
        #fig1 = m.plot(forecast)
        
        return res_json
    
        

if __name__ == '__main__':
    server.run(host= '0.0.0.0', debug=True)
