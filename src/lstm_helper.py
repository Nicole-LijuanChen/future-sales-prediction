import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM

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

def split_to_tain_valid(df,output_length, shift):
    '''
    split into train and valid datasets
    '''




def window_generator(X, y, time_steps=1):
    '''
    Efficiently generate batches of these windows from the training and test data
    Split windows of features into a (features, labels) pairs
    reshape X to [samples, time_steps, n_features]
    '''
    input, output = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        input.append(v)
        output.append(y.iloc[i + time_steps])
    return np.array(input),np.array(output)

def lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(
            units=128,
            input_shape=(X_train.shape[1], X_train.shape[2])
            ))
    model.add(Dense(units=X_train.shape[2]*2))
    model.add(Dense(units=X_train.shape[2]))
    model.compile(
                loss='mse',
                optimizer='Adam') 
    return model

        