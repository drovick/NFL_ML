import pickle

def multitree_get_errors(index,importance_list,error_list,tree_list,iterss,subss,ratess,n_estimators_list,maxdep_list,minsamp_leaf_list,maxfeat_list,minimp_dec_list):
    print('Model:',str(tree_list[index]))
    print('n_estimators:', str(n_estimators_list[index]),', max_depth:', str(maxdep_list[index]),)
    print('min_samples_child:', str(minsamp_leaf_list[index]),', max_features:,', str(maxfeat_list[index]))
    print('min_impurity_decrease:',str(minimp_dec_list[index]),', n_iterations:',str(iterss[index]))
    print('subsample:',str(subss[index]),', learning rate:',str(ratess[index]))
    return error_list[index]

filename = 'xgboost_test'
infile = open(filename,'rb')

i_list = pickle.load(infile)
e_list = pickle.load(infile)
t_list = pickle.load(infile)
tols = pickle.load(infile)
iters = pickle.load(infile)
fracs = pickle.load(infile)
subs = pickle.load(infile)
rates = pickle.load(infile)
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
    print(multitree_get_errors(i,i_list,e_list,t_list,iters,subs,rates,estimators,maxdeps,minsamps,maxfeats,minimp_decs))