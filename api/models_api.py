from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np


app = FastAPI()

class PredictionInput(BaseModel):
    product_id: int
    unit_price: float
    quantity: float
    discount: float
    order_year: int
    order_month: int
    order_day: int

with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.post("/predict")
def predict_sales(input_data: PredictionInput):

    data = np.array([[input_data.product_id, input_data.unit_price, input_data.quantity,
                       input_data.discount, input_data.order_year, input_data.order_month, input_data.order_day]])
    

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)
    
    return {"predicted_total_price": prediction[0]}
