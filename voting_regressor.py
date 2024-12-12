import pandas as pd 
import numpy as np 
import pickle 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
import lightgbm as lgb 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

data = pd.read_csv("data.csv")

train, test = train_test_split(data, test_size=0.15, random_state=0, stratify=data[['label']])

x_train = train.drop(columns=['label'])
x_test = test.drop(columns=['label'])

y_train = train['label']
y_test = test['label']

rf_grid = {'max_depth':range(4,15),
            'n_estimators':range(8,25),
            'min_samples_split': range(2,10),
            'min_samples_leaf': range(4,10),
            'max_features': [0.75,0.8,0.95]}
xgb_grid = 
lgb_grid = 

rf_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1),rf_grid,cv=10,scoring=['neg_root_mean_squared_error','r2'],refit='neg_root_mean_squared_error')
rf_model.fit(x_train,y_train)
print("Best Score:{:,.2f} ", rf_model.best_score_)
print("Best Parameters: ", rf_model.best_params_)
print("Best Estimator: ", rf_model.best_estimator_)