import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time
import hyperopt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

warnings.simplefilter(action='ignore', category=FutureWarning)

db = pd.read_csv(r'/Users/henry/Desktop/train.csv')

# Set the number of data
n_data = 188318

# Converting the loss to log(loss) to improve accuracy
db['loss'] = np.log(db['loss'])

cont_X = db.iloc[0:n_data, 117:131]
cat_X = db.iloc[0:n_data,1:117]
   
X = db.iloc[0:n_data, :131]
Y = db.iloc[0:n_data, 131:]

categorical_features_indices = np.where(X.dtypes != np.float)[0]

# Convert categorical data to continous data
#array = pd.get_dummies(cat_X)

# Merge continuous data and converted categorical data
#new_X = np.c_[array, cont_X]

seed = 7
test_size = 0.2

# Initialise a timer
start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(cont_X, Y, test_size=test_size, random_state=seed)

def objective(space):
    print(space)
    model = CatBoostRegressor(n_estimators=100,
                            depth = int(space['depth']),
                            subsample = space['subsample'],
                            bagging_temperature = space['bagging_temperature'],
                            rsm = space['rsm'],
                            l2_leaf_reg = space['l2_leaf_reg'])

                            
    model.fit(X_train,y_train,cat_features=None,eval_set=(X_test, y_test))

    y_pred = model.predict(X_test)
    MAE = mean_absolute_error(np.exp(y_test), np.exp(y_pred))

    #change the metric if you like
    return {'loss':MAE, 'status': STATUS_OK }

space ={'depth': hp.uniform('x_depth',4,8),
        'bagging_temperature': hp.uniform('x_bagging_temperature',0,1),
        'subsample': hp.uniform('x_subsample', 0.5, 1),
        'rsm': hp.uniform('x_rsm', 0.75, 1.0),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
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
