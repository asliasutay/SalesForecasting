import pandas as pd
from . import db

def check_df(dataframe, head=5):
    print("##################### info #####################")
    dataframe.info()
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Head #####################")
    print(dataframe.head(head))

def orders_df():
    df = db.get_orders()
    print("### Orders DataFrame ###")
    check_df(df)
    
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col])
    print("\n### Tarih Dönüşümünden Sonra Orders DataFrame ###")
    check_df(df)
    
    s_o_distance = df["shipped_date"] - df["order_date"]
    s_o_mean = s_o_distance.mean()
    print("Ortalama süre:", s_o_mean)
    s_o_mean_days = int(s_o_mean / pd.Timedelta(days=1))
    print("Toplam süre:", s_o_distance.sum(), "Ortalama gün:", s_o_mean_days)
    
    null_shipped = df["shipped_date"].isnull()
    print("Null shipped_date sayısı:", null_shipped.sum())
    df.loc[null_shipped, "shipped_date"] = df.loc[null_shipped, "order_date"] + pd.to_timedelta(s_o_mean_days, unit="D")
    print("Güncellenmiş null shipped_date sayısı:", df["shipped_date"].isnull().sum())
    
    if "ship_region" in df.columns:
        df.drop("ship_region", axis=1, inplace=True)
    print("Güncel sütunlar:", df.columns)
    
    if "ship_postal_code" in df.columns:
        df.loc[df["ship_postal_code"].isnull(), "ship_postal_code"] = "11111"
        print("Güncellenmiş ship_postal_code null sayısı:", df["ship_postal_code"].isnull().sum())
    
    return df
