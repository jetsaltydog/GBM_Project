import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import hyperopt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

warnings.simplefilter(action='ignore', category=FutureWarning)

db = pd.read_csv(r'/Users/henry/Desktop/train.csv')

# Set the number of data
n_data = 118318

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

X_train, X_test, y_train, y_test = train_test_split(array, Y, test_size=test_size, random_state=seed)

def objective(space):
    print(space)
    model = xgb.XGBRegressor(objective ='reg:squarederror',
                            n_estimators =100,
                            colsample_bytree=space['colsample_bytree'],
                            learning_rate = 0.3,
                            max_depth = int(space['max_depth']),
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            gamma = space['gamma'],
                            reg_lambda = space['reg_lambda'],)


    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    MAE = mean_absolute_error(np.exp(y_test), np.exp(y_pred))

    #change the metric if you like
    return {'loss':MAE, 'status': STATUS_OK }

space ={'max_depth': hp.quniform("x_max_depth", 4, 16, 1),
        'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),
        'subsample': hp.uniform ('x_subsample', 0.7, 1),
        'gamma' : hp.uniform ('x_gamma', 0.1,0.5),
        'colsample_bytree' : hp.uniform ('x_colsample_bytree', 0.7,1),
        'reg_lambda' : hp.uniform ('x_reg_lambda', 0,1)
    }

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best)
Time = time.time() - start_time
print(str(Time) + " seconds")
