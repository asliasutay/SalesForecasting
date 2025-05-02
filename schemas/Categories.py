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

def categories_df():
   
    df = db.get_categories()
    print("### Categories DataFrame ###")
    check_df(df)
    
    
    if "picture" in df.columns:
        df.drop("picture", axis=1, inplace=True)
        print("\n'picture' kolonu kaldırıldı.\n")
    
    print("### Categories DataFrame ###")
    check_df(df)
    
    return df


