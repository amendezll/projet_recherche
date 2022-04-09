# functions.py>


# The inputs are two vectors of the same size, where y denotes the sales 
# of a store and y_ denotes the corresponding predictions

def MAPE(y, y_) :

    import numpy as np
    
    y = np.array(y)
    y_ = np.array(y_)
    n = len(y)
    return round((1/n)*np.sum((abs(y-y_)/y)), 5)

def MSE(y, y_) :

    import numpy as np
    
    y = np.array(y)
    y_ = np.array(y_)
    n = len(y)
    return round((1/n)*np.sum(((y-y_))**2), 5)

def MAE(y, y_) :

    import numpy as np
    
    y = np.array(y)
    y_ = np.array(y_)
    n = len(y)
    return round((1/n)*np.sum((abs(y-y_))), 5)