import numpy as np
class LinearRegression:
    def __init__(self,alpha=0.01,epochs=1000 ):
        self.alpha=alpha
        self.epochs=epochs
        self.W=None
        self.mean=None
        self.std=None


    def prepare_features(self, X):
        m,n=X.shape
        X_norm = (X - self.mean) / self.std
        one = np.ones((m,1))
        return np.hstack((X_norm, one))
       
    def fit(self,X,Y):
        m,n=X.shape
        self.mean=np.mean(X,axis=0)
        self.std=np.std(X,axis=0)
        # avoid division by zero
        self.std[self.std == 0] = 1
        # Normalising X
        X_norm=self.prepare_features(X)
        #initialising Weighs
        self.W=np.zeros(X.shape[1]+1)
        #gradient descent
        for epoch in range(self.epochs):
            self.W=self.W-self.alpha*(1/m)*np.dot(X_norm.T,(np.dot(X_norm,self.W)-Y))


    
    def compute_cost(self,X,Y):
        if self.W is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        m=X.shape[0]
        X_norm=self.prepare_features(X)
        prediction_error=np.dot(X_norm,self.W)-Y
        return (1/(2*m))*np.dot(prediction_error.T,prediction_error)
 
    
    
    def predict(self,X_input):
        if self.W is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        X_norm=self.prepare_features(X_input)
        return np.dot(X_norm,self.W)
