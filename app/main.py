import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import schemas.Categories as cat
import schemas.Customers as cus
import schemas.OrderDetails as od
import schemas.Orders as o
import schemas.Products as p
from EDA.features_engineering import feature_engineering_pipeline


orders_df = o.orders_df()
order_details_df = od.order_details_df()
customers_df = cus.customers_df()
products_df = p.products_df()
categories_df = cat.categories_df()


all_dataframe = feature_engineering_pipeline(orders_df, order_details_df, customers_df, products_df, categories_df)

print("Birleştirilmiş ve özellik mühendisliği uygulanmış veri:")
print(all_dataframe["merged_df"].head())
