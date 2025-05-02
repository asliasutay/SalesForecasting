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

def order_details_df():
    df = db.get_order_details()
    print("### Order Details DataFrame ###")
    check_df(df)
    return df
