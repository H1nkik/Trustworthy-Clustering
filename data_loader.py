import scipy.io as scio
import numpy as np

def load_data(name):
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['X']
    X = X.astype(np.float32)
    X /= np.max(X)
    return X, labels
