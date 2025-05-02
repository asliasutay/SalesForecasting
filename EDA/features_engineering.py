import pandas as pd

def compute_monthly_sales(orders_df, order_details_df, products_df):
    """
        Aylık Ürün Bazlı Satış Özetleri
    """
    df_temp = pd.merge(order_details_df, orders_df, on="order_id", how="inner")
    df_temp["order_date"] = pd.to_datetime(df_temp["order_date"])
    df_temp["year"] = df_temp["order_date"].dt.year
    df_temp["month"] = df_temp["order_date"].dt.month
    
    monthly_sales = df_temp.groupby(["year", "month", "product_id"]).agg(
        total_quantity=("quantity", "sum"),
        total_revenue=("unit_price", lambda x: (x * df_temp.loc[x.index, "quantity"]).sum())
    ).reset_index()
    
    monthly_sales = pd.merge(monthly_sales, products_df[["product_id", "product_name"]], on="product_id", how="left")
    return monthly_sales

def add_order_date_features(orders_df):
    """
        Orders Tablosu için ay, yıl, gün bilgisi çıkarımı. 
    """
    orders_df["order_year"] = orders_df["order_date"].dt.year
    orders_df["order_month"] = orders_df["order_date"].dt.month
    orders_df["order_day"] = orders_df["order_date"].dt.day
    return orders_df

def calculate_total_price(order_details_df):
    """
        OrderDetails toplam fiyat hesaplaması
    """
    order_details_df["total_price"] = order_details_df["quantity"] * order_details_df["unit_price"] * (1 - order_details_df["discount"])
    return order_details_df

def compute_customer_avg_order_value(orders_df, order_details_df):
    """
        Her müşterinin ortalama harcama tutarını hesaplama
    """
    customer_merge = pd.merge(order_details_df, orders_df, on="order_id", how="inner")
    customer_avg = customer_merge.groupby("customer_id")["total_price"].mean().reset_index()
    customer_avg.rename(columns={"total_price": "AvgOrderValue"}, inplace=True)
    return customer_avg

def update_customers_with_order_count(customers_df, orders_df):
    """
    Orders tablosunda her müşterinin sipariş sayısını hesaplar
    """
    customer_order_count = orders_df['customer_id'].value_counts().reset_index()
    customer_order_count.columns = ['customer_id', 'order_count']
    updated_customers = pd.merge(customers_df, customer_order_count, on='customer_id', how='left')
    updated_customers['order_count'].fillna(0, inplace=True)
    return updated_customers

def segment_customers(customers_df):
    """
    Müşterilerin sipariş sayılarına göre segmentasyon yapar.
    """
    q25 = customers_df['order_count'].quantile(0.25)
    q75 = customers_df['order_count'].quantile(0.75)
    
    def dynamic_segment(row):
        if row['order_count'] >= q75:
            return 'Loyal'
        elif row['order_count'] > q25:
            return 'Regular'
        else:
            return 'New'
    
    customers_df['customer_segment'] = customers_df.apply(dynamic_segment, axis=1)
    return customers_df

def segment_products(products_df):
    """
    Ürünlerin unit_price değerlerine göre segmentasyon yapar.
    """
    def price_segment(row):
        if row['unit_price'] < 20:
            return 'Economy'
        elif row['unit_price'] < 55:
            return 'Standard'
        else:
            return 'Luxury'
    
    products_df['price_segment'] = products_df.apply(price_segment, axis=1)
    return products_df

def compute_product_sales(orders_df, order_details_df, products_df):
    """
        Ürün bazında toplam satış (total_price) hesaplar ve ürün bilgilerini ekler.
    """
    orders_subset = orders_df[['order_id', 'order_month', 'order_year']]
    combined_df = pd.merge(order_details_df, orders_subset, on='order_id', how='inner')
    product_sales = combined_df.groupby(['product_id', 'order_month', 'order_year'])['total_price'].sum().reset_index()
    product_sales = pd.merge(product_sales, products_df[['product_id', 'product_name', 'unit_price']], on='product_id', how='left')
    return product_sales


def merge_all_datasets(orders_df, order_details_df, customers_df, products_df, categories_df, customer_avg_order_value):
    """
    OrderDetails, Orders, Customers, Products ve Categories tablolarını uygun anahtarlar üzerinden birleştirir.
    """
    customers_df = pd.merge(customers_df, customer_avg_order_value, on="customer_id", how="left")
    
    merged_df = pd.merge(order_details_df, orders_df, on="order_id", how="inner")
    merged_df = pd.merge(merged_df, customers_df, on="customer_id", how="left")
    merged_df = pd.merge(merged_df, products_df, on="product_id", how="left")
    merged_df = pd.merge(merged_df, categories_df, on="category_id", how="left")
    return merged_df

def feature_engineering_pipeline(orders_df, order_details_df, customers_df, products_df, categories_df):
    """ 
      - merged_df: Birleşik veri seti.
      - monthly_sales: Aylık satış özetleri.
      - customer_avg_order_value: Müşteri başına ortalama harcama.
      - product_sales: Ürün bazında satış özetleri.
      - customers_df: Güncellenmiş müşteri verisi (sipariş sayısı ve segment bilgisi).
      - products_df: Güncellenmiş ürün verisi (ürün segmentasyonu).
    """
    orders_df = add_order_date_features(orders_df)
    order_details_df = calculate_total_price(order_details_df)
    monthly_sales = compute_monthly_sales(orders_df, order_details_df, products_df)
    customers_df = update_customers_with_order_count(customers_df, orders_df)
    customers_df = segment_customers(customers_df)
    products_df = segment_products(products_df)
    customer_avg_order_value = compute_customer_avg_order_value(orders_df, order_details_df)
    product_sales = compute_product_sales(orders_df, order_details_df, products_df)
    

    merged_df = merge_all_datasets(orders_df, order_details_df, customers_df, products_df, categories_df, customer_avg_order_value)
    
    if 'unit_price_x' in merged_df.columns:
        merged_df.rename(columns={'unit_price_x': 'unit_price'}, inplace=True)

    orders_df.to_csv(r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\orders.csv", index=False)
    order_details_df.to_csv(r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\order_details.csv", index=False)
    monthly_sales.to_csv(r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\monthly_sales.csv", index=False)
    customers_df.to_csv(r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\customers_df.csv", index=False)
    products_df.to_csv(r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\products_df.csv", index=False)
    customer_avg_order_value.to_csv(r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\customer_avg_order_value.csv", index=False)
    product_sales.to_csv(r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\product_sales.csv", index=False)
    merged_df.to_csv(r"C:\Users\zeyne\Desktop\turkcell\Projects\SalesForecasting\data\merged_df.csv", index=False)

    return {
        "merged_df": merged_df,
        "monthly_sales": monthly_sales,
        "customer_avg_order_value": customer_avg_order_value,
        "product_sales": product_sales,
        "customers_df": customers_df,
        "products_df": products_df
    }