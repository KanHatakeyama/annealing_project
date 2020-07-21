
from sklearn import linear_model

def get_model(x,y):
    eta0=1/x.shape[1]*5

    model=linear_model.SGDRegressor(max_iter=1000, 
                                    verbose=1,   
                                    penalty="l2",
                                    alpha=0.1,
                                    eta0=eta0,   #should be small enough for successful learning
                                    l1_ratio=0,
                                    early_stopping=True,
                                    n_iter_no_change=50,  #should be large enough for successful learning
                                )

    model.fit(x,y)
    return model