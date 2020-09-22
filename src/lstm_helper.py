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


def load_data(path):
    '''
    load processed store-item sales data to a dataframe
    '''
    df = pd.read_csv(path)
    #set datatime to index
    df['date'] =  pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

def prepare_data(df,output_cols):
    #load store-itenm sales values
    prep_df = df[output_cols]
    # convert prep_df to a dataframe
    prep_df = pd.DataFrame(prep_df)
    return prep_df

def split_data(df,output_length, shift):
    '''
    split raw dataset into training, validation, and test sets.
    df : raw dataset
    output_length: the number days we would like to predict
    shift: recurrent cell numbers
    '''
    train_size = df.shape[0] - (output_length+shift)
    test_size  = df.shape[0] - train_size
    train = df.iloc[:train_size,:]
    test  = df.iloc[-test_size:,:]
    valid = df.iloc[-output_length:,:]
    return train, test, valid






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

def lstm_model_1(X_train,lstm_units):
    model = Sequential()
    model.add(LSTM(
            units = lstm_units,
            input_shape=(X_train.shape[1], X_train.shape[2])
            ))
    model.add(Dense(units=X_train.shape[2]*2))
    model.add(Dense(units=X_train.shape[2]))
    model.compile(
                loss='mse',
                optimizer='Adam') 
    return model

def lstm_model_10(X_train,lstm_units):
    model = Sequential()
    model.add(LSTM(
            units = lstm_units,
            input_shape=(X_train.shape[1], X_train.shape[2]),
            return_sequences=False
            ))
    model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    model.add(Dense(units=X_train.shape[2]*3))
    model.add(Dense(units=X_train.shape[2]))
    model.compile(
                loss='mse',
                optimizer="rmsprop") 
    return model

def predict_sequence_full(self, data, window_size):
    # Shift the window by 1 new prediction each time,
    # re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame,
                                   [window_size-1],
                                   predicted[-1],
                                   axis=0)
        return predicted


if __name__ == '__main__':
    path ='../data/lstm_data.csv'
    df = load_data(path)
    print(df.shape)
    # extract columns names
    columns = df.columns.tolist()
    total_labels = len(columns)
    # print(f'total labels: {total_labels}')

    # set 10-output LSTM model parameters
    output_length =92 # the number days we would like to predict
    #time_stepts in LSTM: the recurrent cell gets unrolled to a specified length 
    time_steps = 14    #recurrent cell numbers,two weeks
    labels_width =10
    output_length =92
    lstm_units = 128*2

    #creat a datafreme to store the prediction values
    df_fc_500 = pd.DataFrame(index=df.index[-output_length:])
     
    
    for i in range(10, 21,10): 
        # split into training, validation, and test sets.
        df_10 = df.iloc[:,i-10:i]
        output_cols = df_10.columns.tolist()
        train,test,valid = split_data(df_10,output_length, time_steps)
        
        X_train, y_train = window_generator(train, train.iloc[:,:len(output_cols)],time_steps)
        X_test,  y_test  = window_generator(test, test.iloc[:,:len(output_cols)],time_steps)
        
        
        model = lstm_model_10(X_train,lstm_units)
        history = model.fit(
                    X_train, y_train,
                    epochs=30,
                    batch_size=30,
                    validation_split=0.1,
                    verbose=1,
                    shuffle=False)
        y_pred = model.predict(X_test)
        df_forecast = pd.DataFrame(y_pred, index=valid.index, columns=valid.columns + '_forecast')
        df_fc_500 =pd.concat([df_fc_500, df_forecast], axis = 'columns')
        
    print(df_fc_500.shape)
    # df_fc_500.to_csv('../data/lstm_10_predictions_test.csv')  
    print('results saved to csv')

