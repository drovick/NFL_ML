import pickle

def multitree_get_errors(index,importance_list,error_list,tree_list,n_estimators_list,maxdep_list,minsamp_split_list,minsamp_leaf_list,minweight_frac_leaf_list,maxfeat_list,maxleaf_nodes_list,minimp_dec_list):
    print('Model:',str(tree_list[index]),', n_estimators:', str(n_estimators_list[index]),', max_depth:', str(maxdep_list[index]), ', min_samples_split:', str(minsamp_split_list[index]))
    print('min_samples_leaf:', str(minsamp_leaf_list[index]), 'min_weight_fraction_leaf:', str(minweight_frac_leaf_list[index]), ', max_features:,', str(maxfeat_list[index]))
    print('max_leaf_nodes:', str(maxleaf_nodes_list[index]), ', min_impurity_decrease:', str(minimp_dec_list[index]))
    return error_list[index]

filename = 'extratrees_pickle_test'
infile = open(filename,'rb')

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

for i in range(0,len(e_list)):
    print(multitree_get_errors(0,i_list,e_list,t_list,estimators,maxdeps,minsamps,minsamps1,minweights,maxfeats,maxleafs,minimp_decs))