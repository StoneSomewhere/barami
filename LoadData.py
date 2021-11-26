import numpy as np
from WavletTransform import wavlet_transform


def load_data(name, s, wavlet):
    X = np.load('X_%s.npy'%name)
    Y = np.load('Y_%s.npy'%name)
    
            
    if wavlet == True:
        for column in X.columns:
            X[column] = wavlet_transform(X[column])    
        
    if s == 0: return X, Y
        
    #나누기
    X_b = X[-s:]
    Y_b = Y[-s:]

    X_f = X[:-s]
    Y_f = Y[:-s]
        
        

    return X_f, Y_f, X_b, Y_b

