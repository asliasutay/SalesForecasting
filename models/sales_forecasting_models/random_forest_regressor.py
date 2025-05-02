import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from common import scale_features, visualize_regression_performance

def train_random_forest_regression(df_model, features, target, test_size=0.2, random_state=42, n_estimators=100):
    X = df_model[features]
    y = df_model[target]

    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Random Forest Regressor - R2 Score:", r2)
    print("RMSE:", rmse)
    
    return model, scaler, X_test, y_test, y_pred

if __name__ == "__main__":

    df_model = pd.read_csv(r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\merged_df.csv")
    features = ['product_id', 'unit_price', 'quantity', 'discount', 'order_year', 'order_month', 'order_day']
    target = 'total_price'
    
    model, scaler, X_test, y_test, y_pred = train_random_forest_regression(df_model, features, target)
    
    save_path = r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\images\random_forest_performance.png"
    visualize_regression_performance(y_test, y_pred, algorithm_name="Random Forest ", save_path=save_path)
