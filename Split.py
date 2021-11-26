import numpy as np


def load_data(name):

    X = np.load('X_train_%s.npy' % name)
    Y = np.load('Y_train_%s.npy' % name)

    s = len(Y)//2
    # 나누기

    np.save('X_train_%s_0.npy' % name, X[:s])
    np.save('X_train_%s_1.npy' % name, X[s:])
    np.save('Y_train_%s_0.npy' % name, Y[:s])
    np.save('Y_train_%s_1.npy' % name, Y[s:])


load_data("1h_2")
print("Done!")
load_data("1d_2")
print("Done!")
load_data("30m_bit_8")
print("Done!")