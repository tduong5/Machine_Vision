# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Bank_Predictions.csv') # dataframe from pandas

# ------ Part-1: Data preprocessing ----------

# data[from: to]. convert pandas df to numpy array
# loc gets rows (or columns) with particular labels from the index.
# iloc gets rows (or columns) at particular positions in the index (so it only takes integers).
x = dataset.iloc[:, :-1].values # index just the last column by specifying the -1 index
y = dataset.iloc[:, -1].values
    # x contains features, y contains classes

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# labelencoder 1D data, ordinalencoder 2D data
x_labelencoder = LabelEncoder() # to convert this kind of categorical text data into model-understandable numerical data
    ## we can’t have text in our data if we’re going to run any kind of model on it
    ## Encode target labels with value between 0 and n_classes-1.
x[:, 0] = x_labelencoder.fit_transform(x[:, 0])
    # What one hot encoding does is, it takes a column which has categorical data, 
    # which has been label encoded, and then splits the column into multiple columns
hot_encoder = OneHotEncoder() # Encode categorical features as a one-hot numeric array.
x = hot_encoder.fit_transform(x).toarray()

y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(y)

# Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) 
    # test_size = 0.2 mean 20% of data. 
    # if you don’t pass anything, the RandomState instance used by np.random will be used instead.


# Feature Scaling

# normalizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # it is mainly required in case the dataset features vary a lot
x_test = scaler.transform(x_test)

# ------- Part-2: Build the ANN --------

# import keras library and packages
from tensorflow import keras
from tensorflow.keras.models import Sequential # used to initialize neural networks
from tensorflow.keras.layers import Dense # is used to specify the fully connected layer or hidden layers
from tensorflow.keras.layers import Dropout # prevents overfitting

# Initializing the ANN
classifier = Sequential() #  classifier, because the output it will return is a 0 or 1

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    # units - how many neurons or units the connected (hidden) layer will have
    # kern - statistical distribution or function to use for initializing the weights
    # activation - reLU (rectifier linear unit). most commonly used
classifier.add(Dropout(0.1)) 

# Adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    # input_dim already defined. no need to use again
classifier.add(Dropout(0.1))

# Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')
    # Adam is an optimization algorithm that can be used 
    # instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.
    # loss = ‘binary_crossentropy’ to get closer to the loss. output is either a 0 or 1 so binary
    # metrics = 'accuracy' to measure how accurate the NN will be. 

# Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)
    # if 100 training ex's; batch size 10 then 10 iterations to complete 1 epochs

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
print(y_pred)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix # to see how many values we got right or wrong
c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)
""" 
A confusion matrix is a table that is often used to describe the performance of a classification 
model (or “classifier”) on a set of test data for which the true values are known. 
It allows the visualization of the performance of an algorithm.
"""

# evaluate the keras model
accuracy = classifier.evaluate(x, y) # Returns the loss value & metrics values for the model in test mode.
print(accuracy)
