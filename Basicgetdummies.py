import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold

warnings.simplefilter(action='ignore', category=FutureWarning)

db = pd.read_csv(r'/Users/henry/Desktop/train.csv')

n_data = 30000

cont_X = db.iloc[0:n_data, 117:131]
cat_X = db.iloc[0:n_data,1:117]
   
X = db.iloc[0:n_data, :131]
Y = db.iloc[0:n_data, 131:]

#cat_features = list(train.select_dtypes(include=['object']).columns)
#train_d = train.drop(['id','loss'],axis=1)
#Y = train.filter(['loss'])

#train_d = pd.get_dummies(train_d,columns=cat_features)

array = pd.get_dummies(cat_X)

new_X = np.c_[array, cont_X]

seed = 7
test_size = 0.2

start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(array, Y, test_size=test_size, random_state=seed)

model = XGBRegressor(objective ='reg:squarederror')
model.fit(X_train,y_train)
##
#make predictions for test data
y_pred = model.predict(X_test)
predictions = [value for value in y_pred]

#Computing the MAE of our predictions
print("Mean Absolute Error : " + str(mean_absolute_error(y_test, y_pred)))

#Time taken
Time = time.time() - start_time
print(str(Time) + " seconds")

