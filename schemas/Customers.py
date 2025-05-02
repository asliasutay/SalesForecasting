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

def customers_df():
  
    df = db.get_customers()
    print("### Customers DataFrame ###")
    check_df(df)
    
    df.drop(columns=["region", "phone", "fax"], axis=1, inplace=True)
    
    df.loc[df["country"] == "Ireland", "postal_code"] = "11111"
    
    print("\n### Customers DataFrame ###")
    check_df(df)
    
    return df

