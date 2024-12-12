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
xgb_grid = {'booster': ['gblinear','gbtree'],
            'max_depth': range(2,8),
            'alpha': [0.01,0.1,1,1.1,1.01],
            'subsample': [0.75,0.8,0.95]}
lgb_grid = {'num_leaves':[6,10,20],
            'min_child_samples': [2,5,8,10],
            'subsample': [0.75,0.8,0.95],
            'colsample_bytree': [0.4,0.75,0.9,1]}

rf_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1),rf_grid,cv=10,scoring=['neg_root_mean_squared_error','r2'],refit='neg_root_mean_squared_error')
rf_model.fit(x_train,y_train)
print("Best Score:{:,.2f} ", rf_model.best_score_)
print("Best Parameters: ", rf_model.best_params_)
print("Best Estimator: ", rf_model.best_estimator_)

xgb_model = RandomizedSearchCV(XGBRegressor(n_jobs=-1),xgb_grid,cv=10,scoring=['neg_root_mean_squared_error','r2'],refit='neg_root_mean_squared_error')
xgb_model.fit(x_train,y_train)
print("Best Score:{:,.2f} ", xgb_model.best_score_)
print("Best Parameters: ", xgb_model.best_params_)
print("Best Estimator: ", xgb_model.best_estimator_)

lgb_model = RandomizedSearchCV(lgb.LGBMRegressor(n_jobs=-1),lgb_grid,cv=10,scoring=['neg_root_mean_squared_error','r2'],refit='neg_root_mean_squared_error')
lgb_model.fit(x_train,y_train)
print("Best Score:{:,.2f} ", lgb_model.best_score_)
print("Best Parameters: ", lgb_model.best_params_)
print("Best Estimator: ", lgb_model.best_estimator_)

voting_model = VotingRegressor([('rf',rf_model), ('xgb',xgb_model),('lgb',lgb_model)],weights=(0.5,0.25,0.25),n_jobs=-1)
voting_model.fit(x_train,y_train)
print("Best Score:{:,.2f} ", voting_model.best_score_)
print("Best Parameters: ", voting_model.best_params_)
print("Best Estimator: ", voting_model.best_estimator_)