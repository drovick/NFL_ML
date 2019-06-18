import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import numpy as np

import xgboost as xgb
import numpy as np
import time

dataset = pd.read_csv("final_dataset.csv", low_memory=False, parse_dates=['Date'], infer_datetime_format=True)

dataset.drop(columns='Previous_D_Punting_Blck',inplace=True)
dataset.drop(columns=['Unnamed: 0', 'C', 'CB', 'DB', 'DE', 'DL', 'DT', 'G', 'LB', 'LS', 'LT',
        'NT', 'OG', 'OL', 'OT', 'P', 'T'],  inplace=True)

test_set = pd.DataFrame(dataset[dataset.Date>=datetime.datetime(2018,1,1)])
train_set = pd.DataFrame(dataset[dataset.Date<datetime.datetime(2018,1,1)])

test_set.reset_index(inplace=True)
test_set.drop(columns='index', inplace=True)
test_set.set_index(keys=['Name','Date','Tm'],drop=True,append=True,inplace=True,verify_integrity=False)

train_set.set_index(keys=['Name','Date','Tm'],drop=True,append=True,inplace=True,verify_integrity=False)

output_cols = ['Fumbles_Fmb','Kick Returns_TD','Passing_Int','Passing_TD','Passing_Yds','Punt Returns_TD','Receiving_Rec','Receiving_TD','Receiving_Yds','Rushing_TD','Rushing_Yds','Scoring_2PM','Scoring_FGM','Scoring_XPM','Scoring_FG_miss','Scoring_XP_miss','WLT','Team_Pts_for','Team_Pts_against','Team_Pts_diff']
input_cols = set(train_set.columns.values) - set(output_cols)
input_cols.remove('Scoring_Pts')
input_cols.remove('Scoring_Sfty')
input_cols.remove('Scoring_TD')
input_cols = list(input_cols)

train_set_input = train_set[input_cols]
train_set_output = train_set[output_cols]
test_set_input = test_set[input_cols]
test_set_output = test_set[output_cols]

def normalize_input(input_frame,cat_cols,means,std_devs):
    normalized = ((input_frame.drop(columns=cat_cols)-means.drop(labels=cat_cols))/std_devs.drop(labels=cat_cols)).copy()
    cat = input_frame[cat_cols].copy()
    return pd.concat([cat,normalized],axis=1,sort=False)

input_means = train_set_input.mean()
input_std_deviations = train_set_input.std()
categorical_ish = ['Home','Games_GS','Previous_Games_GS','Previous_WLT','Previous_Home','K','QB','TE','WR','RB','FB']

train_set_input_normalized = normalize_input(train_set_input,categorical_ish,input_means,input_std_deviations)
test_set_input_normalized = normalize_input(test_set_input,categorical_ish,input_means,input_std_deviations)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

def br_multiout_error_frame(test_predictions,test_out,label_cols,multiout_strategy='raw_values'):
      
  lin_mse = mean_squared_error(test_out,test_predictions,multioutput=multiout_strategy)
  
  lin_rmse = pd.Series(np.sqrt(lin_mse))
  lin_mae = pd.Series(mean_absolute_error(test_out,test_predictions,multioutput=multiout_strategy))
  lin_evs = pd.Series(explained_variance_score(test_out,test_predictions,multioutput=multiout_strategy))
  lin_r2 = pd.Series(r2_score(test_out,test_predictions,multioutput=multiout_strategy))
  
  frame = pd.concat([lin_r2,lin_evs,lin_mae,lin_rmse],axis=1,sort=False)
  cols = ['R2Score','ExpVarScore','MeanAbsltError','RMSError']
  frame.columns = cols
  
  return frame

def br_singleout_error_frame(test_predictions,test_out,label_cols):
  
  lin_medae = median_absolute_error(test_out.values,test_predictions)
  frame = pd.DataFrame([lin_medae])
  cols = ['MedAbsError']
  frame.columns = cols
  
  return frame

def linear_regression_error_frame(test_predictions,test_out,label_cols):
  
  multiout = br_multiout_error_frame(test_predictions,test_out,label_cols)
  singleout = br_singleout_error_frame(test_predictions,test_out,label_cols)
  
  return pd.concat([singleout,multiout],ignore_index=False,sort=False,axis=1)

num_round = 100

param = {'objective': 'multi:softmax', # Specify multiclass classification
         'num_class': 8, # Number of possible output classes
         'tree_method': 'gpu_hist' # Use GPU accelerated algorithm
         }

dtrain = xgb.DMatrix(train_set_input_normalized, train_set_output['Passing_Yds'])
dtest = xgb.DMatrix(test_set_input_normalized, test_set_output['Passing_Yds'])


gpu_res = {}
tmp = time.time()

# Train model
xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))

# Repeat for CPU algorithm
tmp = time.time()
param['tree_method'] = 'hist'
cpu_res = {}
xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=cpu_res)
print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))