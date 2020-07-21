import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import random
import matplotlib.pyplot as plt

def split_dataframe(df,test_ratio):
    """
    split dataframe

    Parameters
    -----------------
    df: dataframe
       dataframe to be split 
    test_ratio: float
        0-1
    Returns
    -----------------
    train_df: dataframe
        train dataset
    test_df: dataframe
        test dataset
    """
    random.seed(0)
    data_size=df.shape[0]
    test_cases=random.sample(list(range(data_size)),int(data_size*test_ratio))
    train_cases=list(set(list(range(data_size)))-set(test_cases))

    test_cases=[True if i in test_cases else False for i in range(data_size)]
    train_cases=[True if i in train_cases else False for i in range(data_size)]

    train_df=df[train_cases]
    test_df=df[test_cases]
    
    return train_df,test_df

def auto_evaluation(model,x_train,y_train,x_test,y_test):
    """
    auto evaluation of the model

    Parameters
    ----------------
    model: sklearn model
        preditction model 
    x_train: np array
        -
    y_train: np array
        -
    x_test: np array
        -
    y_test: np array
        -
    Returns
    -----------
    y_train_prediction: np array
        predicted y by the model
    y_test_prediction: np array
        predicted y by the model
    """

    y_train_prediction=model.predict(x_train)
    y_test_prediction=model.predict(x_test)

    plt.scatter(y_train,y_train_prediction,c="b",s=1,alpha=0.5)
    plt.scatter(y_test,y_test_prediction,c="r",s=2,alpha=0.5)
    plt.xlabel("actual")
    plt.ylabel("predicted")

    print("tr R2: {:.2f}".format(r2_score(y_train_prediction,y_train)))
    print("te R2: {:.2f}".format(r2_score(y_test_prediction,y_test)))   
    
    return y_train_prediction,y_test_prediction