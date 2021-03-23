import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('insurance.csv')

X=dataset.iloc[:,[0,1,2,4]].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

X[:,1]=le.fit_transform(X[:,1])
X[:,3]=le.fit_transform(X[:,3])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=0)

#X_train[:,[0,2]]=sc_x.fit_transform(X_train[:,[0,2]])
#X_test[:,[0,2]]=sc_x.transform(X_test[:,[0,2]])

#y_train=sc_y.fit_transform(y)
#y_test=sc_y.transform(y)

def LinReg(x,y):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x,y)
    return  regressor

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
def PolyReg(x,y):
    from sklearn.linear_model import LinearRegression
    x_poly = poly_reg.fit_transform(x)
    regressor = LinearRegression()
    regressor.fit(x_poly,y)
    return regressor

def RandForest(x,y):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(x,y)
    return regressor

def DecTrees(x,y):
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(x,y)
    return regressor

from sklearn.metrics import r2_score

linReg=LinReg(X_train,y_train)
ypred_lr=linReg.predict(X_test)
r2_lr=r2_score(y_test,ypred_lr)

polyReg=PolyReg(X_train,y_train)
ypred_pr=polyReg.predict(poly_reg.fit_transform(X_test))
r2_pr=r2_score(y_test,ypred_pr)

decReg=DecTrees(X_train,y_train)
ypred_dt=decReg.predict(X_test)
r2_dt=r2_score(y_test,ypred_dt)

rfReg=RandForest(X_train,y_train)
ypred_rf=rfReg.predict(X_test)
r2_rf=r2_score(y_test,ypred_rf)



