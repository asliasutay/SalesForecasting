import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from common import visualize_confusion_matrix, visualize_class_distribution, print_classification_report_custom

def train_logistic_regression_classifier(csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    required_columns = ["order_count", "AvgOrderValue", "customer_segment"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' sütunu CSV dosyanızda bulunamadı.")
        
    X = df[["order_count", "AvgOrderValue"]]
    y_raw = df["customer_segment"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Logistic Regression Accuracy:", acc)
    print_classification_report_custom(y_test, y_pred, encoder.classes_.tolist())

    with open("logistic_regression_customer.pkl", "wb") as f:
        pickle.dump((model, scaler, encoder), f)
    print("Model saved as logistic_regression_customer.pkl")
    
    class_names = encoder.classes_.tolist()
    visualize_confusion_matrix(y_test, y_pred, class_names, algorithm_name="Logistic Regression", save_path=r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\images\logistic_confusion_matrix.png")
    visualize_class_distribution(y_test, y_pred, class_names, algorithm_name="Logistic Regression", save_path=r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\images\logistic_class_distribution.png")
    return model, scaler, encoder

if __name__ == "__main__":
    csv_path = r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\merged_df.csv"
    train_logistic_regression_classifier(csv_path)
