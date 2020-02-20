import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)

# Read the data
db = pd.read_csv(r'/Users/henry/Desktop/train.csv')

# Set the number of data
n_data = 100

# Converting the loss to log(loss) to improve accuracy
db['loss'] = np.log(db['loss'])

# Select the continuous & categorical data
cont_X = db.iloc[0:n_data, 117:131]
cat_X = db.iloc[0:n_data,1:117]

# Select all the input & output (loss) data   
X = db.iloc[0:n_data, :131]
Y = db.iloc[0:n_data, 131:]

# Convert categorical data to continous data
array = pd.get_dummies(cat_X)

# Merge continuous data and converted categorical data
new_X = np.c_[array, cont_X]

# Set seed in order to produce same results each time
seed = 7

# Set size of test data
test_size = 0.2

# Initialise a timer
start_time = time.time()

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(new_X, Y, test_size=test_size, random_state=seed)

y_train = y_train.iloc[:,0]

# Create the model
model = LGBMRegressor()
model.fit(X_train,y_train)

# Make predictions for test data
y_pred = model.predict(X_test)

# Computing the MAE of our predictions
print("Mean Absolute Error : " + str(mean_absolute_error(np.exp(y_test), np.exp(y_pred))))

# Time taken
Time = time.time() - start_time
print(str(Time) + " seconds")
