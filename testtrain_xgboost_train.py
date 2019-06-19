import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import numpy as np
import xgboost as xgb
import numpy as np
import time
import pickle
import sklearn

dataset = pd.read_csv("final_dataset.csv", low_memory=False, parse_dates=['Date'], infer_datetime_format=True)

dataset.drop(columns='Previous_D_Punting_Blck',inplace=True)
dataset.drop(columns=['Unnamed: 0', 'C', 'CB', 'DB', 'DE', 'DL', 'DT', 'G', 'LB', 'LS', 'LT',
        'NT', 'OG', 'OL', 'OT', 'P', 'T'],  inplace=True)

"""
test_set = pd.DataFrame(dataset[(dataset.Date>=datetime.datetime(2018,1,1))])
val_set = pd.DataFrame(dataset[(dataset.Date>=datetime.datetime(2017,1,1))&(dataset.Date<datetime.datetime(2018,1,1))])
train_set = pd.DataFrame(dataset[(dataset.Date<datetime.datetime(2017,1,1))])
"""
full_set = pd.DataFrame(dataset)


"""
test_set.reset_index(inplace=True)
test_set.drop(columns='index', inplace=True)
test_set.set_index(keys=['Name','Date','Tm'],drop=True,append=True,inplace=True,verify_integrity=False)

val_set.reset_index(inplace=True)
val_set.drop(columns='index', inplace=True)
val_set.set_index(keys=['Name','Date','Tm'],drop=True,append=True,inplace=True,verify_integrity=False)
"""
full_set.reset_index(inplace=True)
full_set.drop(columns='index', inplace=True)
full_set.set_index(keys=['Name','Date','Tm'],drop=True,append=True,inplace=True,verify_integrity=False)


#train_set.set_index(keys=['Name','Date','Tm'],drop=True,append=True,inplace=True,verify_integrity=False)


output_cols = ['Fumbles_Fmb','Kick Returns_TD','Passing_Int','Passing_TD','Passing_Yds','Punt Returns_TD','Receiving_Rec','Receiving_TD','Receiving_Yds','Rushing_TD','Rushing_Yds','Scoring_2PM','Scoring_FGM','Scoring_XPM','Scoring_FG_miss','Scoring_XP_miss','WLT','Team_Pts_for','Team_Pts_against','Team_Pts_diff']
#input_cols = set(train_set.columns.values) - set(output_cols)
input_cols = set(full_set.columns) - set(output_cols)
input_cols.remove('Scoring_Pts')
input_cols.remove('Scoring_Sfty')
input_cols.remove('Scoring_TD')
input_cols = list(input_cols)

"""
test_set_input = test_set[input_cols]
test_set_output = test_set[output_cols]

val_set_input = val_set[input_cols]
val_set_output = val_set[output_cols]

train_set_input = train_set[input_cols]
train_set_output = train_set[output_cols]
"""

full_set_input = full_set[input_cols]
full_set_output = full_set[output_cols]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(full_set_input, full_set_output, test_size=0.2)
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.2)


"""
def normalize_input(input_frame,cat_cols,means,std_devs):
    normalized = ((input_frame.drop(columns=cat_cols)-means.drop(labels=cat_cols))/std_devs.drop(labels=cat_cols)).copy()
    cat = input_frame[cat_cols].copy()
    return pd.concat([cat,normalized],axis=1,sort=False)
"""

def normalize_input(input_frame,cat_cols,means,std_devs):
    normalized = ((input_frame.drop(columns=cat_cols)-means.drop(labels=cat_cols))/std_devs.drop(labels=cat_cols)).copy()
    cat = input_frame[cat_cols].copy()
    return pd.concat([cat,normalized],axis=1,sort=False)

"""

input_means = train_set_input.mean()
input_std_deviations = train_set_input.std()
categorical_ish = ['Home','Games_GS','Previous_Games_GS','Previous_WLT','Previous_Home','K','QB','TE','WR','RB','FB']

"""
input_means = pd.DataFrame(X_test, columns=input_cols).mean()
input_std_deviations = pd.DataFrame(X_val,columns=input_cols).std()
categorical_ish = ['Home','Games_GS','Previous_Games_GS','Previous_WLT','Previous_Home','K','QB','TE','WR','RB','FB']
"""
test_set_input_normalized = normalize_input(test_set_input,categorical_ish,input_means,input_std_deviations)
val_set_input_normalized = normalize_input(val_set_input,categorical_ish,input_means,input_std_deviations)
train_set_input_normalized = normalize_input(train_set_input,categorical_ish,input_means,input_std_deviations)

"""
test_set_input_normalized = normalize_input(pd.DataFrame(X_test, columns=input_cols),categorical_ish,input_means,input_std_deviations)
val_set_input_normalized = normalize_input(pd.DataFrame(X_val,columns=input_cols),categorical_ish,input_means,input_std_deviations)
train_set_input_normalized = normalize_input(pd.DataFrame(X_train,columns=input_cols),categorical_ish,input_means,input_std_deviations)

test_set_output = pd.DataFrame(y_test, columns=output_cols)
val_set_output = pd.DataFrame(y_val, columns=output_cols)
train_set_output = pd.DataFrame(y_train, columns=output_cols)


###DATA PREPARING ENDS

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

def gbr_multitree_fit_errors(model,train_in,train_out,test_in,test_out,input_cols_,output_cols_,n_iter_no_change_=None,subsample_=0.1,learning_rate_=0.1,n_estimators_=50,max_depth_=20,min_samples_leaf_=1,max_features_=1.0,min_impurity_decrease_=0):
    
    # Convert test data from numpy to XGBoost format
    xgbtest = xgb.DMatrix(test_in, label=test_out)
    
    params = { 'booster' : 'gbtree',
        'tree_method' : 'gpu_hist', # Use GPU accelerated algorithm
        'predictor' : 'gpu_predictor',
        'objective' : 'reg:squarederror',
        'evals' : [(xgbtest, 'test')]
        }
    
    tree_model = xgb.XGBRegressor(early_stopping_rounds=n_iter_no_change_,max_depth=max_depth_,subsample=subsample_,learning_rate=learning_rate_,n_estimators=n_estimators_,min_child_weight=min_samples_leaf_,colsample_bytree=max_features_,gamma=min_impurity_decrease_,**params)
     
    tree_model.fit(train_in,train_out)
    
    test_predictions = tree_model.predict(test_in)
    imps = tree_model.feature_importances_ 
    
    return pd.DataFrame(tree_model.feature_importances_,index=input_cols).rename(index=str,columns={0:'Importance'}).sort_values('Importance',ascending=False),linear_regression_error_frame(test_predictions,test_out,output_cols_)
    #return   imps,linear_regression_error_frame(test_predictions,test_out,output_cols_)

def gbr_multitree_loop_lin_results(models,train_in,train_out,test_in,test_out,input_cols_,output_cols_,n_iter_no_change__,subsample,learning_rate,n_estimators,max_depths,min_samples_leafs,max_featuress,min_impurity_decreases):
    features_list=[]
    error_list=[]
    importances_list=[]
     
    tree_list=[]
    iters_list=[]

    sub_list=[]
    rate_list=[]
    n_estimators_list=[]
    maxdep_list=[]
    minsamp_leaf_list=[]
    maxfeat_list=[]
    minimp_dec_list=[]
    
    i=0
  
    for mod in models:
        #for iters_ in n_iter_no_change__:
        for sub in subsample:
            for rate in learning_rate:
                for n_ in n_estimators:
                    for depth in max_depths:
                        for minsamp_leaf in min_samples_leafs:
                            for maxfeat in max_featuress:
                                for minimp_dec in min_impurity_decreases:
                                        
                                    tree_list.append(str(mod))                                                            
                                    #iters_list.append(iters_)
                                    iters_list.append(n_iter_no_change__)
                                    sub_list.append(sub)
                                    rate_list.append(rate)   
                                    n_estimators_list.append(n_)
                                    maxdep_list.append(depth)
                                    minsamp_leaf_list.append(minsamp_leaf)
                                    maxfeat_list.append(maxfeat)
                                    minimp_dec_list.append(minimp_dec)
                                        
                                    error_frame = pd.DataFrame()
                                    importances_frame = pd.DataFrame()
                                                            
                                    print('entering multitree_fit_errors')
                                    c = 0
                                    for col in output_cols_:
                                        c+=1
                                        importances,error = gbr_multitree_fit_errors(mod,train_in,train_out[col],test_in,test_out[col],input_cols,col,n_iter_no_change_=n_iter_no_change__,subsample_=sub,learning_rate_=rate,n_estimators_=n_,max_depth_=depth,min_samples_leaf_=minsamp_leaf,max_features_=maxfeat,min_impurity_decrease_=minimp_dec)               
                                        if c == 10:
                                            print('model trained for 10 output features, 10 more to go..')
                                        error_frame = pd.concat([error_frame,error],axis=0)
                                        importances.index = input_cols_            
                                        importances_frame = pd.concat([importances_frame,importances],axis=1,ignore_index=False)                                                            
                                                                                                                                                                                   
                                    importances_list.append(importances_frame)                                               
                                    error_list.append(error_frame)            
                                                            

                                        
                                    print('tree trained! index:', str(i))
                                    i +=1

    return importances_list,error_list,tree_list,iters_list,sub_list,rate_list,n_estimators_list,maxdep_list,minsamp_leaf_list,maxfeat_list,minimp_dec_list

###Training starts here..

print('about to start training the first group..')
tmp = time.time()

i_list,e_list,t_list,iters,subs,rates,estimators,maxdeps,minsamps,maxfeats,minimp_decs = gbr_multitree_loop_lin_results(['XGBRegressor'],train_set_input_normalized,train_set_output,train_set_input_normalized,train_set_output,input_cols,output_cols,n_iter_no_change__=+5,subsample=[float(+1.0)],learning_rate=[float(+0.1)],n_estimators=[50],max_depths=[20],min_samples_leafs=[2],max_featuress=[0.2],min_impurity_decreases=[float(+0.01),float(+0.001)])
print('trained the first group, GPU Training Time: %s seconds'% (str(time.time() - tmp)))


print('will pickle and save it to a file before proceeding..')  
print(str(len(e_list)), ' models trained and evaluated, attempting to pickle..')
import pickle
filename = 'testtrain_xgbpickle_1'
outfile = open(filename,'wb')
pickle_objs = [i_list,e_list,t_list,iters,subs,rates,estimators,maxdeps,minsamps,maxfeats,minimp_decs]
for obj in pickle_objs:
  pickle.dump(obj,outfile)
outfile.close()
print('pickling complete, will now train the second group of models')


tmp = time.time()
print('about to start training the second group..')

i,e,t,it,su,ra,estimat,maxd,minsa,maxfe,minidecs = gbr_multitree_loop_lin_results(['XGBRegressor'],train_set_input_normalized,train_set_output,train_set_input_normalized,train_set_output,input_cols,output_cols,n_iter_no_change__=+10,subsample=[float(+0.1),float(+0.01)],learning_rate=[float(+0.01),float(0.001)],n_estimators=[80],max_depths=[50],min_samples_leafs=[2],max_featuress=[0.4,0.6],min_impurity_decreases=[float(+0.01),float(+0.001)])
print('trained the second group, GPU Training Time: %s seconds'% (str(time.time() - tmp)))
print('will now append to lists..')     

i_list.extend(i)
e_list.extend(e)
t_list.extend(t)
iters.extend(it)
subs.extend(su)
rates.extend(ra)
estimators.extend(estimat)
maxdeps.extend(maxd)
minsamps.extend(minsa)
maxfeats.extend(maxfe)
minimp_decs.extend(minidecs)

print('append succesful, ',str(len(e_list)), ' models trained and evaluated, attempting to pickle..')

filename = 'testtrain_xgbpickle_2'
outfile = open(filename,'wb')
pickle_objs = [i_list,e_list,t_list,iters,subs,rates,estimators,maxdeps,minsamps,maxfeats,minimp_decs]

for obj in pickle_objs:
  pickle.dump(obj,outfile)

outfile.close()

print('pickling complete, training the final group..')

i,e,t,it,su,ra,estimat,maxd,minsa,maxfe,minidecs = gbr_multitree_loop_lin_results(['XGBRegressor'],train_set_input_normalized,train_set_output,train_set_input_normalized,train_set_output,input_cols,output_cols,n_iter_no_change__=+10,subsample=[float(+0.1),float(+0.01)],learning_rate=[float(+0.01),float(0.001)],n_estimators=[80],max_depths=[50,75,100],min_samples_leafs=[2],max_featuress=[0.7,0.8,1.0],min_impurity_decreases=[float(+0.005),float(+0.001)])
print('trained the final group, will now append to list structures and pickle..')

i_list.extend(i)
e_list.extend(e)
t_list.extend(t)
iters.extend(it)
subs.extend(su)
rates.extend(ra)
estimators.extend(estimat)
maxdeps.extend(maxd)
minsamps.extend(minsa)
maxfeats.extend(maxfe)
minimp_decs.extend(minidecs)

print('append succesful, ',str(len(e_list)), ' models trained and evaluated, attempting to pickle..')

filename = 'testtrain_xgb_pickle'
outfile = open(filename,'wb')

for obj in pickle_objs:
  pickle.dump(obj,outfile)

outfile.close()

print('pickling complete! check file:',str(filename))
print('EOF!')