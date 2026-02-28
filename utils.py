import numpy as np

def train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=None):
    X=np.array(X)
    Y=np.array(Y)
    n_samples=X.shape[0]
    n_train=int(n_samples*(1-test_size))
    indices=np.arange(n_samples)
    if shuffle:
        rng=np.random.default_rng(random_state)
        rng.shuffle(indices)
    train_indices=indices[:n_train]
    test_indices=indices[n_train:]
    X_test=X[test_indices]
    X_train=X[train_indices]
    Y_test=Y[test_indices]
    Y_train=Y[train_indices]
    return X_train, X_test, Y_train, Y_test



def r2_score(y_true,y_pred):
    mean=np.mean(y_true,axis=0)
    SS_res=np.dot((y_true-y_pred).T,(y_true-y_pred))
    SS_tot=np.dot((y_true-mean).T,(y_true-mean))
    return 1-SS_res/SS_tot