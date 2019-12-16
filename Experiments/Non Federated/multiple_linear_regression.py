# Multiple Linear Regression

# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('student-mat.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 9].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# dmy Var - trpap if they are highly correlated - use the dummy 
# X = X[:, 1:]
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_test_pred = regressor.predict(X_test)
y_train_pred = regressor.predict(X_train)
#print( " this is the predicted  output ******************************")
#print(y_pred)



#print("this is the actual data")
data =X_train[:,0:1]

plt.plot(y_train, y_train_pred,'*r')
plt.plot(y_test, y_test_pred, '*g')
plt.figure()

yTrainr2 = (r2_score(y_train, y_train_pred))
YTest = (r2_score(y_test, y_test_pred))


