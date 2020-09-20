import pandas as pd
import numpy as np

# path ='data/lstm_data.csv'

def process_product_group(df,groups,prefixs):
    """
    feature engineering:
        get dummies on categorical
    """
    processed_df = df.copy()
    for i, group in enumerate(groups):
    #Get the Dummy variables
        dummies = pd.get_dummies(df[group], prefix=prefixs[i],drop_first = True)
        processed_df = processed_df.join(dummies)
    return processed_df

def data_pipeline(path):
    df = pd.read_csv(path)
    cat_groups =['store','item']
    prefixs    =['s','i']
    processed_df =   process_product_group(df,cat_groups,prefixs)
    
    # drop_cols =['store','item','year','year_and_month']
    drop_cols =['year','year_and_month']

    processed_df = processed_df.drop(columns =drop_cols)

    return processed_df

# processed_df = data_pipeline(path)

# print(processed_df.shape)
# print(processed_df.head())
