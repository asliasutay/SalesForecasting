import pandas as pd
from sqlalchemy import create_engine


DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "GYK1Northwind"

db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)

def get_orders():
    dfOrders = pd.read_sql("SELECT * FROM ORDERS", engine)
    print("\ndfOrders\n", dfOrders.head(3))
    return dfOrders

def get_order_details():
    dfOrderDetails = pd.read_sql("SELECT * FROM ORDER_DETAILS", engine)
    print("\ndfOrderDetails\n", dfOrderDetails.head(3))
    return dfOrderDetails

def get_products():
    dfProducts = pd.read_sql("SELECT * FROM PRODUCTS", engine)
    print("\ndfProducts\n", dfProducts.head(3))
    return dfProducts

def get_customers():
    dfCustomers = pd.read_sql("SELECT * FROM CUSTOMERS", engine)
    print("\ndfCustomers\n", dfCustomers.head(3))
    return dfCustomers

def get_categories():
    dfCategories = pd.read_sql("SELECT * FROM Categories", engine)
    print("\ndfCategories\n", dfCategories.head(3))
    return dfCategories

