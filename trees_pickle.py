import pickle
import sys

def compare_errors(e_lst):
    best_model_index = 0
    best_error = 500
    output_cols = ['Fumbles_Fmb','Kick Returns_TD','Passing_Int','Passing_TD','Passing_Yds','Punt Returns_TD','Receiving_Rec','Receiving_TD','Receiving_Yds','Rushing_TD','Rushing_Yds','Scoring_2PM','Scoring_FGM','Scoring_XPM','Scoring_FG_miss','Scoring_XP_miss','WLT','Team_Pts_for','Team_Pts_against','Team_Pts_diff']

  
    for i in range(0,len(e_lst)):
        e_lst[i].index = output_cols
        passing_err = e_lst[i].loc['Passing_Yds','RMSError']
        rushing_err = e_lst[i].loc['Rushing_Yds','RMSError']
        receiving_err = e_lst[i].loc['Receiving_Yds','RMSError']
        error_sum = passing_err + rushing_err + receiving_err
    
        if error_sum < best_error:
            best_error = error_sum
            best_model_index = i
  
    return best_model_index, best_error

def multitree_get_errors(index,importance_list,error_list,tree_list,n_estimators_list,maxdep_list,minsamp_split_list,minsamp_leaf_list,minweight_frac_leaf_list,maxfeat_list,maxleaf_nodes_list,minimp_dec_list):
    print('model index:',str(index),', Model:',str(tree_list[index]),', n_estimators:', str(n_estimators_list[index]),', max_depth:', str(maxdep_list[index]))
    print('min_samples_split:', str(minsamp_split_list[index]),', min_samples_leaf:', str(minsamp_leaf_list[index]), 'min_weight_fraction_leaf:', str(minweight_frac_leaf_list[index]))
    print('max_features:,', str(maxfeat_list[index]),' max_leaf_nodes:', str(maxleaf_nodes_list[index]), ', min_impurity_decrease:', str(minimp_dec_list[index]))
    print(' ')
    return error_list[index]

filename = str(sys.argv[1])
print('opening ',filename,' pickle file..')
infile = open(filename,'rb')
print('unpacking objects..')

i_list = pickle.load(infile)
e_list = pickle.load(infile)
t_list = pickle.load(infile)
estimators = pickle.load(infile)
maxdeps = pickle.load(infile)
minsamps = pickle.load(infile)
minsamps1 = pickle.load(infile)
minweights = pickle.load(infile)
maxfeats = pickle.load(infile)
maxleafs = pickle.load(infile)
minimp_decs = pickle.load(infile)

infile.close()
print('objects unpacked!')
print('results obtained for ', str(len(e_list)),' models, now printing..')

output_cols = ['Fumbles_Fmb','Kick Returns_TD','Passing_Int','Passing_TD','Passing_Yds','Punt Returns_TD','Receiving_Rec','Receiving_TD','Receiving_Yds','Rushing_TD','Rushing_Yds','Scoring_2PM','Scoring_FGM','Scoring_XPM','Scoring_FG_miss','Scoring_XP_miss','WLT','Team_Pts_for','Team_Pts_against','Team_Pts_diff']

for i in range(0,len(e_list)):
    e_list[i].index=output_cols
    print(multitree_get_errors(i,i_list,e_list,t_list,estimators,maxdeps,minsamps,minsamps1,minweights,maxfeats,maxleafs,minimp_decs))
    print(' ')
    
best_m_index, best_e = compare_errors(e_list)


print('#########################################################################################')
print('#########################################################################################')
print(' ')
print('best performing model (rmse) across Passing_Yds, Rushing_Yds and Rec_Yds: ',str(best_m_index)')
print('sum of their RMSEs:',best_e)
print(' ')
print('printing best error frame..')
print(' ')
print(multitree_get_errors(best_m_index,i_list,e_list,t_list,estimators,maxdeps,minsamps,minsamps1,minweights,maxfeats,maxleafs,minimp_decs))
print(' ')
print('#########################################################################################')
print('#########################################################################################')
print(' ')
print('..EOF')