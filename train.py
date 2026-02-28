import numpy as np
import pandas as pd 
from models import LinearRegression
import utils

def main():
    df=pd.read_csv("housing.csv")

    X=df.iloc[:,:-1].values
    Y=df.iloc[:,-1].values

    X_train,X_test,Y_train,Y_test=utils.train_test_split(X,Y)
    # Initialize model
    model=LinearRegression()
    # Train model
    model.fit(X_train,Y_train)

    predictions=model.predict(X_train)

    R2_score=utils.r2_score(Y_test,predictions)

    print("R2_score is:",R2_score)

if __name__ == "__main__":
    main()

