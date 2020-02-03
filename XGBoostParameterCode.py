import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV

warnings.simplefilter(action='ignore', category=FutureWarning)

db = pd.read_csv(r'/Users/henry/Desktop/train.csv')

# Set the number of data
n_data = 1000

# Converting the loss to log(loss) to improve accuracy
db['loss'] = np.log(db['loss'])

cont_X = db.iloc[0:n_data, 117:131]
cat_X = db.iloc[0:n_data,1:117]
   
X = db.iloc[0:n_data, :131]
Y = db.iloc[0:n_data, 131:]

# Convert categorical data to continous data
array = pd.get_dummies(cat_X)

# Merge continuous data and converted categorical data
new_X = np.c_[array, cont_X]

seed = 7
test_size = 0.2

# Initialise a timer
start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(new_X, Y, test_size=test_size, random_state=seed)

# Turn the arrays from (x,1) to (x,)
y_train = np.squeeze(np.array(y_train))
y_test = np.squeeze(np.array(y_test))

# use DMatrix for xgboost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# use svmlight file for xgboost
dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')

# default xgboost params
xgb_params_default = {}

# set xgboost params
xgb_params = {
    'seed': 0, # used for generating reproducible results
    'eta': 0.1, # controls the learning rate
    'gamma': 0, # controls regularization (or prevents overfitting)
    'colsample_bytree': 0.5, # control the number of features (variables) supplied to a tree
    'silent': 1, # logging mode - quiet
    'subsample': 0.5, # controls the number of samples (observations) supplied to a tree.
    'objective': 'reg:linear',
    'max_depth': 5, # determines how deeply each tree is allowed to grow during any boosting round
    'min_child_weight': 3  # minimum number of instances required in a child node
}

#model = XGBRegressor(objective ='reg:squarederror',eta=0.1)
model = XGBRegressor(objective ='reg:squarederror')

# Construct a parameter grid
param_grid = {'max_depth': range(4,9), 'min_child_weight': (1,3,6)}

# Carry out a grid search
gridsearch = GridSearchCV(model,param_grid)

gridsearch.fit(X_train,y_train)

#print(gridsearch.cv_results_)
print(gridsearch.best_estimator_)
print(gridsearch.best_params_)

# Set the best parameters found by the grid search
best_params = gridsearch.best_params_

#model.fit(X_train,y_train)
#y_pred = model.predict(X_test)

# training and testing
### DEFAULT ###
bst = xgb.train(xgb_params_default, dtrain)

### DEFINED ###
#bst = xgb.train(xgb_params, dtrain)

### GRID SEARCH ###
#bst = xgb.train(best_params, dtrain)

# make predictions for test data
preds = bst.predict(dtest)

# dump the model
bst.dump_model('dump.raw.txt')

# Computing the MAE of our predictions
#print("Mean Absolute Error : " + str(mean_absolute_error(preds, y_test)))

# Log version
print("Mean Absolute Error : " + str(mean_absolute_error(np.exp(preds), np.exp(y_test))))

# Time taken
Time = time.time() - start_time
print(str(Time) + " seconds")

#xgb.plot_importance(bst)
#plt.show()
