import numpy as np
import pandas as pd



class SimpleLinearRegression:
    def __init__(self):
        self.theta_0 = 0
        self.theta_1 = 0

    def fit(self, X, y):

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X-x_mean)**(y-y_mean))
        denominator  = np.sum((X-x_mean)**2)

        self.theta_0 = y_mean - self.theta_1*x_mean
        self.theta_1 = numerator/denominator


    def predict(self, X):
        return self.theta_0+self.theta_1*X

class LinearRegressionOSL:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X_bias = np.c_[np.ones(X.shape[0]), X]

        self.theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self,X):

        if self.theta is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit")

        X_bias  = np.c_[np.ones(X.shape[0]), X]

        return X_bias @ self.theta

df = pd.read_csv("BikeData.csv")
numeric_features = [col for col in df.columns if df[col].dtype!='object']

median_temp = df.loc[df["Temperature"].notna(), "Temperature"].median()
df["Temperature"] = df["Temperature"].fillna(median_temp)

X = df[numeric_features].drop(columns=["Temperature"]).values

y = df['Temperature'].values.reshape(-1,1)

model = LinearRegressionOSL()

model.fit(X,y)

y_pred  = model.predict(X)
print(y)
print(y_pred)