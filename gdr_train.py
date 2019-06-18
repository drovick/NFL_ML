import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import numpy as np

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

from sklearn.ensemble import GradientBoostingRegressor

def gbr_multitree_fit_errors(model,train_in,train_out,test_in,test_out,input_cols_,output_cols_,tol_=0.001,n_iter_no_change_=None,validation_fraction_=None,subsample_=0.1,learning_rate_=0.1,n_estimators_=10,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1,min_weight_fraction_leaf_=0,max_features_=None,max_leaf_nodes_=None,min_impurity_decrease_=0):
    
    tree_model = model(tol=tol_,n_iter_no_change=n_iter_no_change_,validation_fraction=validation_fraction_,subsample=subsample_,learning_rate=learning_rate_,n_estimators=n_estimators_,max_depth=max_depth_,min_samples_split=min_samples_split_,min_samples_leaf=min_samples_leaf_,min_weight_fraction_leaf=min_weight_fraction_leaf_,max_features=max_features_,max_leaf_nodes=max_leaf_nodes_,min_impurity_decrease=min_impurity_decrease_)
     
    tree_model.fit(train_in, train_out)
    test_predictions = tree_model.predict(test_in)
    imps = tree_model.feature_importances_ 
    
    return pd.DataFrame(tree_model.feature_importances_,index=input_cols).rename(index=str,columns={0:'Importance'}).sort_values('Importance',ascending=False),linear_regression_error_frame(test_predictions,test_out,output_cols_)
    #return   imps,linear_regression_error_frame(test_predictions,test_out,output_cols_)

def gbr_multitree_loop_lin_results(models,train_in,train_out,test_in,test_out,input_cols_,output_cols_,tol,n_iter_no_change,validation_fraction,subsample,learning_rate,n_estimators,max_depths,min_samples_splits,min_samples_leafs,min_weight_fraction_leafs,max_featuress,max_leaf_nodess,min_impurity_decreases):
    features_list=[]
    error_list=[]
    importances_list=[]
    #model_list=[]
    
    tree_list=[]
    tols_list=[]
    iters_list=[]
    frac_list=[]
    sub_list=[]
    rate_list=[]
    n_estimators_list=[]
    maxdep_list=[]
    minsamp_split_list=[]
    minsamp_leaf_list=[]
    minweight_frac_leaf_list=[]
    maxfeat_list=[]
    maxleaf_nodes_list=[]
    minimp_dec_list=[]
    
    i=0
  
    for mod in models:
        for tols in tol:
            for iters in n_iter_no_change:
                for frac in validation_fraction:
                    for sub in subsample:
                        for rate in learning_rate:
                            for n_estimators in n_estimators:
                                for depth in max_depths:
                                    for minsamp_split in min_samples_splits:
                                        for minsamp_leaf in min_samples_leafs:
                                            for minweight_frac in min_weight_fraction_leafs:
                                                for maxfeat in max_featuress:
                                                    for maxleaf_nodes in max_leaf_nodess:
                                                        for minimp_dec in min_impurity_decreases:

                                                            error_frame = pd.DataFrame()
                                                            importances_frame = pd.DataFrame()
                                                            
                                                            print('entering multitree_fit_errors')
                                                            c = 0
                                                            for col in output_cols_:
                                                                
                                                                c+=1
                                                                importances,error = gbr_multitree_fit_errors(mod,train_in,train_out[col],test_in,test_out[col],input_cols,col,tol_=tols,n_iter_no_change_=iters,validation_fraction_=frac,subsample_=sub,learning_rate_=rate,n_estimators_=n_estimators,max_depth_=depth,min_samples_split_=minsamp_split,min_samples_leaf_=minsamp_leaf,min_weight_fraction_leaf_=minweight_frac,max_features_=maxfeat,max_leaf_nodes_=maxleaf_nodes,min_impurity_decrease_=minimp_dec)               
                                                                if c == 10:
                                                                    print('model trained for 10 output features, 10 more to go..')
                                                                error_frame = pd.concat([error_frame,error],axis=0)
                                                                importances.index = input_cols_            
                                                                importances_frame = pd.concat([importances_frame,importances],axis=1,ignore_index=False)                                                            
                                                            
                                                            
                                                            
                                                            importances_list.append(importances_frame)                                               
                                                            error_list.append(error_frame)            
                                                            
                                                            tree_list.append(str(mod))                                                            
                                                            tols_list.append(tols)
                                                            iters_list.append(frac)
                                                            frac_list.append(frac)
                                                            sub_list.append(sub)
                                                            rate_list.append(rate)   
                                                            n_estimators_list.append(n_estimators)
                                                            maxdep_list.append(depth)
                                                            minsamp_split_list.append(minsamp_split)
                                                            minsamp_leaf_list.append(minsamp_leaf)
                                                            minweight_frac_leaf_list.append(minweight_frac)
                                                            maxfeat_list.append(maxfeat)
                                                            maxleaf_nodes_list.append(maxleaf_nodes)
                                                            minimp_dec_list.append(minimp_dec)
                                                            print('tree trained! index:', str(i))

                    return importances_list,error_list,tree_list,tols_list,iters_list,frac_list,sub_list,rate_list,n_estimators_list,maxdep_list,minsamp_split_list,minsamp_leaf_list,minweight_frac_leaf_list,maxfeat_list,maxleaf_nodes_list,minimp_dec_list

#i_list,e_list,t_list,tols,iters,fracs,subs,rates,estimators,maxdeps,minsamps,minsamps1,minweights,maxfeats,maxleafs,minimp_decs = gbr_multitree_loop_lin_results([GradientBoostingRegressor],train_set_input_normalized,train_set_output,test_set_input_normalized,test_set_output,input_cols,output_cols,tol=[float(+0.1)],n_iter_no_change=[int(+5)],validation_fraction=[float(+0.1)],subsample=[float(+1.0)],learning_rate=[float(+0.1)],n_estimators=[5],max_depths=[50],min_samples_splits=[2],min_samples_leafs=[1],min_weight_fraction_leafs=[0],max_featuress=[75],max_leaf_nodess=[None],min_impurity_decreases=[float(+0.01)])
#i,e,t,tol,it,fr,su,ra,estimat,maxd,minsa,minsa1,minwei,maxfe,maxlea,minidecs = gbr_multitree_loop_lin_results([GradientBoostingRegressor],train_set_input_normalized,train_set_output,test_set_input_normalized,test_set_output,input_cols,output_cols,tol=[float(+0.1)],n_iter_no_change=[int(+5)],validation_fraction=[float(+0.1)],subsample=[float(+1.0)],learning_rate=[float(+0.1)],n_estimators=[5],max_depths=[20],min_samples_splits=[2],min_samples_leafs=[1],min_weight_fraction_leafs=[0],max_featuress=[75],max_leaf_nodess=[None],min_impurity_decreases=[float(+0.3)])

i_list,e_list,t_list,tols,iters,fracs,subs,rates,estimators,maxdeps,minsamps,minsamps1,minweights,maxfeats,maxleafs,minimp_decs = gbr_multitree_loop_lin_results([GradientBoostingRegressor],train_set_input_normalized,train_set_output,test_set_input_normalized,test_set_output,input_cols,output_cols,tol=[float(+0.01)],n_iter_no_change=[+4],validation_fraction=[float(+0.2)],subsample=[float(+1.0)],learning_rate=[float(+0.1)],n_estimators=[50],max_depths=[20,50],min_samples_splits=[2],min_samples_leafs=[1],min_weight_fraction_leafs=[0],max_featuress=[None],max_leaf_nodess=[None],min_impurity_decreases=[float(+0.005)])
print('trained the first group, will pickle and save it to a file before proceeding..')

print(str(len(e_list)), ' models trained and evaluated, attempting to pickle..')
import pickle
filename = 'gdr_pickle_1'
outfile = open(filename,'wb')
pickle_objs = [i_list,e_list,t_list,tols,iters,fracs,subs,rates,estimators,maxdeps,minsamps,minsamps1,minweights,maxfeats,maxleafs,minimp_decs]
for obj in pickle_objs:
  pickle.dump(obj,outfile)
outfile.close()

print('pickling complete, will now train the second group of models')

i,e,t,tol,it,fr,su,ra,estimat,maxd,minsa,minsa1,minwei,maxfe,maxlea,minidecs = gbr_multitree_loop_lin_results([GradientBoostingRegressor],train_set_input_normalized,train_set_output,test_set_input_normalized,test_set_output,input_cols,output_cols,tol=[float(+0.01)],n_iter_no_change=[int(+4)],validation_fraction=[float(+0.2)],subsample=[float(+0.01)],learning_rate=[float(+0.01),float(+0.001)],n_estimators=[50],max_depths=[40],min_samples_splits=[15],min_samples_leafs=[1],min_weight_fraction_leafs=[0],max_featuress=[None],max_leaf_nodess=[None],min_impurity_decreases=[float(+0.005)])
print('trained the second group, will now append to list structures..')

i_list.extend(i)
e_list.extend(e)
t_list.extend(t)
tols.extend(tol)
iters.extend(it)
fracs.extend(fr)
subs.extend(su)
rates.extend(ra)
estimators.extend(estimat)
maxdeps.extend(maxd)
minsamps.extend(minsa)
minsamps1.extend(minsa1)
minweights.extend(minwei)
maxfeats.extend(maxfe)
maxleafs.extend(maxlea)
minimp_decs.extend(minidecs)

print('append succesful, ',str(len(e_list)), ' models trained and evaluated, attempting to pickle..')

filename = 'gdr_pickle_2'
outfile = open(filename,'wb')
pickle_objs = [i_list,e_list,t_list,tols,iters,fracs,subs,rates,estimators,maxdeps,minsamps,minsamps1,minweights,maxfeats,maxleafs,minimp_decs]

for obj in pickle_objs:
  pickle.dump(obj,outfile)

outfile.close()
print('pickling complete, training the final group..')

i,e,t,tol,it,fr,su,ra,estimat,maxd,minsa,minsa1,minwei,maxfe,maxlea,minidecs = gbr_multitree_loop_lin_results([GradientBoostingRegressor],train_set_input_normalized,train_set_output,test_set_input_normalized,test_set_output,input_cols,output_cols,tol=[float(+0.01)],n_iter_no_change=[int(+4)],validation_fraction=[float(+0.2)],subsample=[float(+0.005)],learning_rate=[float(+0.001)],n_estimators=[50],max_depths=[20,50],min_samples_splits=[15],min_samples_leafs=[1],min_weight_fraction_leafs=[0],max_featuress=[None],max_leaf_nodess=[None],min_impurity_decreases=[float(+0.005)])
print('trained the final group, will now append to list structures and pickle..')

i_list.extend(i)
e_list.extend(e)
t_list.extend(t)
tols.extend(tol)
iters.extend(it)
fracs.extend(fr)
subs.extend(su)
rates.extend(ra)
estimators.extend(estimat)
maxdeps.extend(maxd)
minsamps.extend(minsa)
minsamps1.extend(minsa1)
minweights.extend(minwei)
maxfeats.extend(maxfe)
maxleafs.extend(maxlea)
minimp_decs.extend(minidecs)

print('append succesful, ',str(len(e_list)), ' models trained and evaluated, attempting to pickle..')

filename = 'gdr_pickle'
outfile = open(filename,'wb')
pickle_objs = [i_list,e_list,t_list,tols,iters,fracs,subs,rates,estimators,maxdeps,minsamps,minsamps1,minweights,maxfeats,maxleafs,minimp_decs]

for obj in pickle_objs:
  pickle.dump(obj,outfile)

outfile.close()

print('pickling complete! EOF!')