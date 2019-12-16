import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# source - git hub repository https://github.com/antaloaalonso/Regression-Model-YT-Video/blob/master/Regression%20Model%20on%20YT%20Videos%20Dataset%20.ipynb
# author :
scaler = StandardScaler()
df1 = pd.read_csv('student2.csv')

print('printing x values ')
x = df1.iloc[:,:-1].values
y = df1.iloc[:,1].values


# in the fuuture - will do a better way to do cross validation i could underfit my data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# Runs model (the one with the activation function, although this doesn't really matter as they perform the same)
# with its current weights on the training and testing data
# todo change the featuring scaling
X_train =scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# todo add more layers and more parameters
def create_complied_keras_model():
    # regression modelsklearn
    model = Sequential()
    model.add(Dense(5, input_shape=(1,), activation = 'relu'))
    # model.add(Dense(1, kernel_initializer='normal'))
    # model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(Adam(lr=10000), 'mean_squared_error')
    return model

# def create_compiled_keras_model():
#   model = tf.keras.models.Sequential([
#       tf.keras.layers.Dense(
#           10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(8,))])
#
#   model.compile(
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#       optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
#       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#   return model

model = create_complied_keras_model()
# train
# history = model.fit(X_train,y_train,epochs= 500,validation_split = 0.1 , verbose =0)
# history_dict = history.history
model.fit(X_train, y_train, epochs=500)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


plt.plot(y_train, y_train_pred,'*r')
plt.plot(y_test, y_test_pred, '*g')
plt.figure()
#
# print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("mean square")
print(mean_squared_error(y_train_pred[:195],y[:195]))
