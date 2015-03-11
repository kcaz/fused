import numpy as np
import fused_reg as fl
import data_sources as ds
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import collections
#SECTION: ------------------UTILITY FUNCTIONS-----------------




#SECTION: ------------------FOR RUNNING BACTERIAL DATA



def fit_model(data_fn, lamP, lamR, lamS, solver='solve_ortho_direct',special_args=None):
    ds1 = ds.standard_source(data_fn,0)
    ds2 = ds.standard_source(data_fn,1)
    orth_fn = os.path.join(data_fn, 'orth')

    organisms = [ds1.name, ds2.name]
    orth = ds.load_orth(orth_fn, organisms)

    
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()

    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()

        # jam things together
    Xs = [t1_tr, t2_tr]
    Ys = [e1_tr, e2_tr]
    
    genes = [genes1, genes2]
    tfs = [tfs1, tfs2]
    priors = priors1 + priors2

    if solver == 'solve_ortho_direct':
        Bs = fl.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)
    if solver == 'solve_ortho_direct_scad':
        Bs = fl.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, s_it = special_args['s_it'], special_args = special_args)
    if solver == 'solve_ortho_ref':
        Bs = fl.solve_ortho_ref(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)
    
    return Bs

#runs the basic model with specified parameters under k-fold cross-validation, and stores a number of metrics
#k: the number of cv folds
#reverse: train on the little dude (reverse train and test)
#cv_both: if false, always use all the data for the corresponding species
#exclude_tfs: don't evaluate on transcription factors. this is useful for generated data, where you can't hope to get them right
def cv_model1(data_fn, lamP, lamR, lamS, k, solver='solve_ortho_direct',special_args=None, reverse=False, cv_both=(True,True), exclude_tfs=True, eval_con=False):
    print special_args
    ds1 = ds.standard_source(data_fn,0)
    ds2 = ds.standard_source(data_fn,1)
    if eval_con:
        (constraints, marks, orth) = ds.load_constraints(data_fn)
    else:
        constraints = None

    orth_fn = os.path.join(data_fn, 'orth')

    organisms = [ds1.name, ds2.name]
    orth = ds.load_orth(orth_fn, organisms)
    #accumulate metrics
    (mse1, mse2, R21, R22, aupr1, aupr2, auroc1, auroc2) = list(np.zeros(8))
    #accumulate squared metrics for variance
    (v_mse1, v_mse2, v_R21, v_R22, v_aupr1, v_aupr2, v_auroc1, v_auroc2) = list(np.zeros(8))
    
    folds1 = ds1.partition_data(k)
    folds2 = ds2.partition_data(k)

    excl = lambda x,i: x[0:i]+x[(i+1):]
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()

    corr_acc = 0.0 #to get average correlation of fused coefficients

    #clear the last_results output 
    file(os.path.join(data_fn, 'last_results'),'w').close()
    for fold in range(k):
        #get conditions for current cross-validation fold
        f1_te_c = folds1[fold]
        f1_tr_c = np.hstack(excl(folds1, fold))

        f2_te_c = folds2[fold]
        f2_tr_c = np.hstack(excl(folds2, fold))
        
        if reverse:
            tmp = f1_tr_c
            f1_tr_c = f1_te_c
            f1_te_c = tmp
            
            tmp = f2_tr_c
            f2_tr_c = f2_te_c
            f2_te_c = tmp
        
        
        #load train and test data
        if cv_both[0]:
            (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data(f1_tr_c)
            (e1_te, t1_te, genes1, tfs1) = ds1.load_data(f1_te_c)
        else:
            (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
            (e1_te, t1_te, genes1, tfs1) = ds1.load_data()
        if cv_both[1]:
            (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data(f2_tr_c)
            (e2_te, t2_te, genes2, tfs2) = ds2.load_data(f2_te_c)
        else:
            (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
            (e2_te, t2_te, genes2, tfs2) = ds2.load_data()

        print 'training set 1: %d, training set 2: %d' % (e1_tr.shape[0], e2_tr.shape[0])
        print 'test set 1: %d, test set 2: %d'% (e1_te.shape[0], e2_te.shape[0])
        
        # jam things together
        Xs = [t1_tr, t2_tr]
        Ys = [e1_tr, e2_tr]
        genes = [genes1, genes2]
        tfs = [tfs1, tfs2]
        priors = priors1 + priors2
        
        #solve the model
        if solver == 'solve_ortho_direct':
            Bs = fl.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)
        if solver == 'solve_ortho_direct_scad':
            Bs = fl.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, s_it = special_args['s_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_mcp':
            Bs = fl.solve_ortho_direct_mcp(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, m_it = special_args['m_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_em':
            Bs = fl.solve_ortho_direct_em(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, em_it = special_args['em_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_mcp_r':
            Bs = fl.solve_ortho_direct_mcp_r(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)

         #get correlation of fused coefficients, for diagnostic purposes
        
        (corr, fused_coeffs) = fused_coeff_corr(organisms, genes, tfs, orth, Bs)
        corr_acc += corr
        

        mse_err1 = prediction_error(t1_te, Bs[0], e1_te, 'mse', exclude_tfs=exclude_tfs)
        mse_err2 = prediction_error(t2_te, Bs[1], e2_te, 'mse', exclude_tfs=exclude_tfs)
        mse1 += mse_err1 
        mse2 += mse_err2
        v_mse1 += mse_err1**2
        v_mse2 += mse_err2**2


        r2_err1 = prediction_error(t1_te, Bs[0], e1_te, 'R2', exclude_tfs=exclude_tfs)
        r2_err2 = prediction_error(t2_te, Bs[1], e2_te, 'R2', exclude_tfs=exclude_tfs)
        R21 += r2_err1
        R22 += r2_err2
        v_R21 += r2_err1**2
        v_R22 += r2_err2**2

        

        if len(priors1):
            aupr1_err = eval_network_pr(Bs[0], genes1, tfs1, priors1, exclude_tfs=exclude_tfs, constraints=constraints,sub=0)
            aupr1 += aupr1_err
            v_aupr1 += aupr1_err**2
 
        if len(priors2):
            aupr2_err = eval_network_pr(Bs[1], genes2, tfs2, priors2, exclude_tfs=exclude_tfs,constraints=constraints,sub=1)
            aupr2 += aupr2_err
            v_aupr2 += aupr2_err**2

        if len(priors1):
            auroc1_err = eval_network_roc(Bs[0], genes1, tfs1, priors1, exclude_tfs=exclude_tfs,constraints=constraints,sub=0)
            auroc1 += auroc1_err
            v_auroc1 += auroc1_err**2
        if len(priors2):        
            auroc2_err = eval_network_roc(Bs[1], genes2, tfs2, priors2, exclude_tfs=exclude_tfs,constraints=constraints,sub=1)
            auroc2 += auroc2_err
            v_auroc2 += auroc2_err**2
        
        print np.array((v_mse1, v_mse2, v_R21, v_R22, v_aupr1, v_aupr2, v_auroc1, v_auroc2))/(fold+1)
    errd = {'mse':(mse1/k, mse2/k), 'R2':(R21/k, R22/k), 'aupr':(aupr1/k, aupr2/k),'auroc':(auroc1/k, auroc2/k)}
    vrrd = {'mse':(v_mse1/k, v_mse2/k), 'R2':(v_R21/k, v_R22/k), 'aupr':(v_aupr1/k, v_aupr2/k),'auroc':(v_auroc1/k, v_auroc2/k)}
            
    for key in errd.keys():
        acc = []
        for i in range(len(errd[key])):
            acc.append(vrrd[key][i]-errd[key][i]**2)
        errd[key + '_v'] = acc
            

    params_str =  solver + '\t' + 'lamP=%f lamR=%f lamS=%f' % (lamP, lamR, lamS)
    
    result_str_l = []
    header_str_l = ['params']
    for k in ['mse','R2','aupr','auroc']:
        kv = k+'_v'
        result_str_l.append(errd[k][0])
        result_str_l.append(errd[kv][0])
        result_str_l.append(errd[k][1])
        result_str_l.append(errd[kv][1])
        header_str_l.append(k+'1')
        header_str_l.append(kv+'1')
        header_str_l.append(k+'2')
        header_str_l.append(kv+'2')
    result_str = '\t'.join(map(str, result_str_l))
    header_str = '\t'.join(map(str, header_str_l))                       
    
    print params_str + '\t' + result_str + '\n'
    print 'correlation of fused coefficients is %f' % corr
    if not os.path.exists(os.path.join(data_fn, 'results')):
        with open(os.path.join(data_fn, 'results'),'w') as outf:
            outf.write(header_str + '\n')
    with open(os.path.join(data_fn, 'results'),'a') as outf:
        outf.write(params_str + '\t' + result_str + '\n')
    with open(os.path.join(data_fn, 'last_results'),'a') as outf:
        outf.write(header_str + '\n')
        outf.write(params_str + '\t' + result_str + '\n')

    
    return errd

#runs the basic model with specified parameters under k-fold cross-validation, and stores a number of metrics
#returns array for plotting in seaborn
#k: the number of cv folds
#reverse: train on the little dude (reverse train and test)
#cv_both: if false, always use all the data for the corresponding species
#exclude_tfs: don't evaluate on transcription factors. this is useful for generated data, where you can't hope to get them right
def cv_model2(data_fn, lamP, lamR, lamS, k, solver='solve_ortho_direct',special_args=None, reverse=False, cv_both=(True,True), exclude_tfs=True):
    errd = cv_model(data_fn, lamP, lamR, lamS, k, solver,special_args, reverse, cv_both, exclude_tfs)
    err_list = []
    err_list.append(errd['mse'][0])
    err_list.append(errd['mse'][1])
    err_list.append(errd['R2'][0])
    err_list.append(errd['R2'][1])
    err_list.append(errd['aupr'][0])
    err_list.append(errd['aupr'][1])
    err_list.append(errd['auroc'][0])
    err_list.append(errd['auroc'][1])
    
    return err_list


#runs the basic model with specified parameters under k-fold cross-validation
#stores a bunch of metrics, applied to each CV-fold
#k: the number of cv folds
#reverse: train on the little dude (reverse train and test)
#cv_both: if false, always use all the data for the corresponding species
#exclude_tfs: don't evaluate on transcription factors. this is useful for generated data, where you can't hope to get them right
#doesn't output any files
def cv_model3(data_fn, lamP, lamR, lamS, k, solver='solve_ortho_direct',special_args=None, reverse=False, cv_both=(True,True), exclude_tfs=True):
    
    ds1 = ds.standard_source(data_fn,0)
    ds2 = ds.standard_source(data_fn,1)
    
    (constraints, marks, orth) = ds.load_constraints(data_fn)
    

    orth_fn = os.path.join(data_fn, 'orth')

    organisms = [ds1.name, ds2.name]
    orth = ds.load_orth(orth_fn, organisms)
    #accumulate metrics
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, 1)) for m in metrics}
    err_dict2 = {m : np.zeros((k, 1)) for m in metrics}
    err_dicts = [err_dict1, err_dict2] #for indexing
    folds1 = ds1.partition_data(k)
    folds2 = ds2.partition_data(k)

    excl = lambda x,i: x[0:i]+x[(i+1):]
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()

    
    for fold in range(k):
        #get conditions for current cross-validation fold
        f1_te_c = folds1[fold]
        f1_tr_c = np.hstack(excl(folds1, fold))

        f2_te_c = folds2[fold]
        f2_tr_c = np.hstack(excl(folds2, fold))
        
        if reverse:
            tmp = f1_tr_c
            f1_tr_c = f1_te_c
            f1_te_c = tmp
            
            tmp = f2_tr_c
            f2_tr_c = f2_te_c
            f2_te_c = tmp
        
        
        #load train and test data
        if cv_both[0]:
            (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data(f1_tr_c)
            (e1_te, t1_te, genes1, tfs1) = ds1.load_data(f1_te_c)
        else:
            (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
            (e1_te, t1_te, genes1, tfs1) = ds1.load_data()
        if cv_both[1]:
            (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data(f2_tr_c)
            (e2_te, t2_te, genes2, tfs2) = ds2.load_data(f2_te_c)
        else:
            (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
            (e2_te, t2_te, genes2, tfs2) = ds2.load_data()
        
        # jam things together
        Xs = [t1_tr, t2_tr]
        Ys = [e1_tr, e2_tr]
        genes = [genes1, genes2]
        tfs = [tfs1, tfs2]
        priors = priors1 + priors2
        
        #solve the model
        if solver == 'solve_ortho_direct':
            Bs = fl.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)
        if solver == 'solve_ortho_direct_scad':
            Bs = fl.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, s_it = special_args['s_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_mcp':
            Bs = fl.solve_ortho_direct_mcp(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, m_it = special_args['m_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_em':
            Bs = fl.solve_ortho_direct_em(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, em_it = special_args['em_it'], special_args=special_args)

         #get correlation of fused coefficients, for diagnostic purposes
        
        (corr, fused_coeffs) = fused_coeff_corr(organisms, genes, tfs, orth, Bs)
        err_dicts[0]['corr'][fold,0] = corr
        err_dicts[1]['corr'][fold,0] = corr
        print 'training set 1: %d, training set 2: %d' % (e1_tr.shape[0], e2_tr.shape[0])
        print 'test set 1: %d, test set 2: %d'% (e1_te.shape[0], e2_te.shape[0])
        
        
        for bi in [0,1]:
            if bi==0:
                t_te = t1_te
                e_te = e1_te
                priors_bi = priors1
                genes_bi = genes1
                tfs_bi = tfs1
            else:
                t_te = t2_te
                e_te = e2_te
                priors_bi = priors2
                genes_bi = genes2
                tfs_bi = tfs2

            mse = prediction_error(t_te, Bs[bi], e_te, 'mse', exclude_tfs=exclude_tfs)
            err_dicts[bi]['mse'][fold, 0] = mse
            mse = prediction_error(t_te, Bs[bi], e_te, 'R2', exclude_tfs=exclude_tfs)
            err_dicts[bi]['R2'][fold, 0] = mse
            if len(priors_bi):
                aupr = eval_network_pr(Bs[bi], genes_bi, tfs_bi, priors_bi, exclude_tfs=exclude_tfs, constraints = None)
                err_dicts[bi]['aupr'][fold,0] = aupr
                aupr_con = eval_network_pr(Bs[bi], genes_bi, tfs_bi, priors_bi, exclude_tfs=exclude_tfs, constraints = constraints, sub=bi)
                err_dicts[bi]['aupr_con'][fold,0] = aupr                
            
                auc = eval_network_roc(Bs[bi], genes_bi, tfs_bi, priors_bi, exclude_tfs = exclude_tfs, constraints= None)
                err_dicts[bi]['auc'][fold,0] = auc
                auc_con = eval_network_roc(Bs[bi], genes_bi, tfs_bi, priors_bi, exclude_tfs = exclude_tfs, constraints= None)
                err_dicts[bi]['auc_con'][fold,0] = auc_con
    return err_dicts

#cv_model3, but with pct_priors
def cv_model4(data_fn, lamP, lamR, lamS, k, solver='solve_ortho_direct',special_args=None, reverse=False, cv_both=(True,True), exclude_tfs=True, pct_priors=0):
    
    ds1 = ds.standard_source(data_fn,0)
    ds2 = ds.standard_source(data_fn,1)
    
    (constraints, marks, orth) = ds.load_constraints(data_fn)
    

    orth_fn = os.path.join(data_fn, 'orth')

    organisms = [ds1.name, ds2.name]
    orth = ds.load_orth(orth_fn, organisms)
    #accumulate metrics
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, 1)) for m in metrics}
    err_dict2 = {m : np.zeros((k, 1)) for m in metrics}
    err_dicts = [err_dict1, err_dict2] #for indexing
    folds1 = ds1.partition_data(k)
    folds2 = ds2.partition_data(k)

    excl = lambda x,i: x[0:i]+x[(i+1):]
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()
    conds1 = np.arange(len(priors1))
    random.shuffle(conds1)
    priors1_tr = []
    priors1_te = []
    amt_priors_tr1 = int(round(pct_priors*len(priors1)))

    priors1_tr = map(lambda i: priors1[i], conds1[0:amt_priors_tr1])
    priors1_te = map(lambda i: priors1[i], conds1[amt_priors_tr1:])

    conds2 = np.arange(len(priors2))
    random.shuffle(conds2)
    priors2_tr = []
    priors2_te = []
    amt_priors_tr2 = int(round(pct_priors*len(priors2)))
    
    priors2_tr = map(lambda i: priors2[i], conds2[0:amt_priors_tr2])
    priors2_te = map(lambda i: priors2[i], conds2[amt_priors_tr2:])


    
    for fold in range(k):
        #get conditions for current cross-validation fold
        f1_te_c = folds1[fold]
        f1_tr_c = np.hstack(excl(folds1, fold))

        f2_te_c = folds2[fold]
        f2_tr_c = np.hstack(excl(folds2, fold))
        
        if reverse:
            tmp = f1_tr_c
            f1_tr_c = f1_te_c
            f1_te_c = tmp
            
            tmp = f2_tr_c
            f2_tr_c = f2_te_c
            f2_te_c = tmp
        
        
        #load train and test data
        if cv_both[0]:
            (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data(f1_tr_c)
            (e1_te, t1_te, genes1, tfs1) = ds1.load_data(f1_te_c)
        else:
            (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
            (e1_te, t1_te, genes1, tfs1) = ds1.load_data()
        if cv_both[1]:
            (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data(f2_tr_c)
            (e2_te, t2_te, genes2, tfs2) = ds2.load_data(f2_te_c)
        else:
            (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
            (e2_te, t2_te, genes2, tfs2) = ds2.load_data()
        
        # jam things together
        Xs = [t1_tr, t2_tr]
        Ys = [e1_tr, e2_tr]
        genes = [genes1, genes2]
        tfs = [tfs1, tfs2]
        priors_tr = priors1_tr + priors2_tr
        
        #solve the model
        if solver == 'solve_ortho_direct':
            Bs = fl.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors_tr, lamP, lamR, lamS)
        if solver == 'solve_ortho_direct_scad':
            Bs = fl.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors_tr, lamP, lamR, lamS, s_it = special_args['s_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_mcp':
            Bs = fl.solve_ortho_direct_mcp(organisms, genes, tfs, Xs, Ys, orth, priors_tr, lamP, lamR, lamS, m_it = special_args['m_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_em':
            Bs = fl.solve_ortho_direct_em(organisms, genes, tfs, Xs, Ys, orth, priors_tr, lamP, lamR, lamS, em_it = special_args['em_it'], special_args=special_args)

         #get correlation of fused coefficients, for diagnostic purposes
        
        (corr, fused_coeffs) = fused_coeff_corr(organisms, genes, tfs, orth, Bs)
        err_dicts[0]['corr'][fold,0] = corr
        err_dicts[1]['corr'][fold,0] = corr
        print 'training set 1: %d, training set 2: %d' % (e1_tr.shape[0], e2_tr.shape[0])
        print 'test set 1: %d, test set 2: %d'% (e1_te.shape[0], e2_te.shape[0])
        
        
        for bi in [0,1]:
            if bi==0:
                t_te = t1_te
                e_te = e1_te
                priors_bi = priors1_te
                genes_bi = genes1
                tfs_bi = tfs1
            else:
                t_te = t2_te
                e_te = e2_te
                priors_bi = priors2_te
                genes_bi = genes2
                tfs_bi = tfs2

            mse = prediction_error(t_te, Bs[bi], e_te, 'mse', exclude_tfs=exclude_tfs)
            err_dicts[bi]['mse'][fold, 0] = mse
            mse = prediction_error(t_te, Bs[bi], e_te, 'R2', exclude_tfs=exclude_tfs)
            err_dicts[bi]['R2'][fold, 0] = mse
            if len(priors_bi):
                aupr = eval_network_pr(Bs[bi], genes_bi, tfs_bi, priors_bi, exclude_tfs=exclude_tfs, constraints = None)
                err_dicts[bi]['aupr'][fold,0] = aupr
                aupr_con = eval_network_pr(Bs[bi], genes_bi, tfs_bi, priors_bi, exclude_tfs=exclude_tfs, constraints = constraints, sub=bi)
                err_dicts[bi]['aupr_con'][fold,0] = aupr                
            
                auc = eval_network_roc(Bs[bi], genes_bi, tfs_bi, priors_bi, exclude_tfs = exclude_tfs, constraints= None)
                err_dicts[bi]['auc'][fold,0] = auc
                auc_con = eval_network_roc(Bs[bi], genes_bi, tfs_bi, priors_bi, exclude_tfs = exclude_tfs, constraints= None)
                err_dicts[bi]['auc_con'][fold,0] = auc_con
    return err_dicts

#cv_model1, but with percent_priors
#runs the basic model with specified parameters under k-fold cross-validation, and stores a number of metrics
#percent_priors is the percent of priors to use. these priors are removed from the test set
#k: the number of cv folds
#reverse: train on the little dude (reverse train and test)
#cv_both: if false, always use all the data for the corresponding species
#exclude_tfs: don't evaluate on transcription factors. this is useful for generated data, where you can't hope to get them right
def cv_model5(data_fn, lamP, lamR, lamS, k, solver='solve_ortho_direct',special_args=None, reverse=False, cv_both=(True,True), exclude_tfs=True, eval_con=False, pct_priors=0):
    print special_args
    ds1 = ds.standard_source(data_fn,0)
    ds2 = ds.standard_source(data_fn,1)
    if eval_con:
        (constraints, marks, orth) = ds.load_constraints(data_fn)
    else:
        constraints = None

    orth_fn = os.path.join(data_fn, 'orth')

    organisms = [ds1.name, ds2.name]
    orth = ds.load_orth(orth_fn, organisms)
    #accumulate metrics
    (mse1, mse2, R21, R22, aupr1, aupr2, auroc1, auroc2) = list(np.zeros(8))
    #accumulate squared metrics for variance
    (v_mse1, v_mse2, v_R21, v_R22, v_aupr1, v_aupr2, v_auroc1, v_auroc2) = list(np.zeros(8))
    
    folds1 = ds1.partition_data(k)
    folds2 = ds2.partition_data(k)

    excl = lambda x,i: x[0:i]+x[(i+1):]
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()

    conds1 = np.arange(len(priors1))
    random.shuffle(conds1)
    amt_priors1 = int(round(pct_priors*len(priors1)))
    priors1_tr = priors1[conds[0:amt_priors1]]
    priors1_te = priors1[conds[amt_priors:len(priors1)-1]]

    conds2 = np.arange(len(priors2))
    random.shuffle(conds2)
    amt_priors2 = int(round(pct_priors*len(priors1)))
    priors2_tr = priors2[conds[0:amt_priors2]]
    priors2_te = priors2[conds[amt_priors:len(priors2)-1]]

    corr_acc = 0.0 #to get average correlation of fused coefficients

    #clear the last_results output 
    file(os.path.join(data_fn, 'last_results'),'w').close()
    for fold in range(k):
        #get conditions for current cross-validation fold
        f1_te_c = folds1[fold]
        f1_tr_c = np.hstack(excl(folds1, fold))

        f2_te_c = folds2[fold]
        f2_tr_c = np.hstack(excl(folds2, fold))
        
        if reverse:
            tmp = f1_tr_c
            f1_tr_c = f1_te_c
            f1_te_c = tmp
            
            tmp = f2_tr_c
            f2_tr_c = f2_te_c
            f2_te_c = tmp
        
        
        #load train and test data
        if cv_both[0]:
            (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data(f1_tr_c)
            (e1_te, t1_te, genes1, tfs1) = ds1.load_data(f1_te_c)
        else:
            (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
            (e1_te, t1_te, genes1, tfs1) = ds1.load_data()
        if cv_both[1]:
            (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data(f2_tr_c)
            (e2_te, t2_te, genes2, tfs2) = ds2.load_data(f2_te_c)
        else:
            (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
            (e2_te, t2_te, genes2, tfs2) = ds2.load_data()

        print 'training set 1: %d, training set 2: %d' % (e1_tr.shape[0], e2_tr.shape[0])
        print 'test set 1: %d, test set 2: %d'% (e1_te.shape[0], e2_te.shape[0])
        
        # jam things together
        Xs = [t1_tr, t2_tr]
        Ys = [e1_tr, e2_tr]
        genes = [genes1, genes2]
        tfs = [tfs1, tfs2]
        priors_te = priors1_te + priors2_te
        
        #solve the model
        if solver == 'solve_ortho_direct':
            Bs = fl.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors_te, lamP, lamR, lamS)
        if solver == 'solve_ortho_direct_scad':
            Bs = fl.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors_te, lamP, lamR, lamS, s_it = special_args['s_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_mcp':
            Bs = fl.solve_ortho_direct_mcp(organisms, genes, tfs, Xs, Ys, orth, priors_te, lamP, lamR, lamS, m_it = special_args['m_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_em':
            Bs = fl.solve_ortho_direct_em(organisms, genes, tfs, Xs, Ys, orth, priors_te, lamP, lamR, lamS, em_it = special_args['em_it'], special_args=special_args)
        if solver == 'solve_ortho_direct_mcp_r':
            Bs = fl.solve_ortho_direct_mcp_r(organisms, genes, tfs, Xs, Ys, orth, priors_te, lamP, lamR, lamS)

         #get correlation of fused coefficients, for diagnostic purposes
        
        (corr, fused_coeffs) = fused_coeff_corr(organisms, genes, tfs, orth, Bs)
        corr_acc += corr
        

        mse_err1 = prediction_error(t1_te, Bs[0], e1_te, 'mse', exclude_tfs=exclude_tfs)
        mse_err2 = prediction_error(t2_te, Bs[1], e2_te, 'mse', exclude_tfs=exclude_tfs)
        mse1 += mse_err1 
        mse2 += mse_err2
        v_mse1 += mse_err1**2
        v_mse2 += mse_err2**2


        r2_err1 = prediction_error(t1_te, Bs[0], e1_te, 'R2', exclude_tfs=exclude_tfs)
        r2_err2 = prediction_error(t2_te, Bs[1], e2_te, 'R2', exclude_tfs=exclude_tfs)
        R21 += r2_err1
        R22 += r2_err2
        v_R21 += r2_err1**2
        v_R22 += r2_err2**2

        

        if len(priors1):
            aupr1_err = eval_network_pr(Bs[0], genes1, tfs1, priors1_te, exclude_tfs=exclude_tfs, constraints=constraints,sub=0)
            aupr1 += aupr1_err
            v_aupr1 += aupr1_err**2
 
        if len(priors2):
            aupr2_err = eval_network_pr(Bs[1], genes2, tfs2, priors2_te, exclude_tfs=exclude_tfs,constraints=constraints,sub=1)
            aupr2 += aupr2_err
            v_aupr2 += aupr2_err**2

        if len(priors1):
            auroc1_err = eval_network_roc(Bs[0], genes1, tfs1, priors1_te, exclude_tfs=exclude_tfs,constraints=constraints,sub=0)
            auroc1 += auroc1_err
            v_auroc1 += auroc1_err**2
        if len(priors2):        
            auroc2_err = eval_network_roc(Bs[1], genes2, tfs2, priors2_te, exclude_tfs=exclude_tfs,constraints=constraints,sub=1)
            auroc2 += auroc2_err
            v_auroc2 += auroc2_err**2
        
        print np.array((v_mse1, v_mse2, v_R21, v_R22, v_aupr1, v_aupr2, v_auroc1, v_auroc2))/(fold+1)
    errd = {'mse':(mse1/k, mse2/k), 'R2':(R21/k, R22/k), 'aupr':(aupr1/k, aupr2/k),'auroc':(auroc1/k, auroc2/k)}
    vrrd = {'mse':(v_mse1/k, v_mse2/k), 'R2':(v_R21/k, v_R22/k), 'aupr':(v_aupr1/k, v_aupr2/k),'auroc':(v_auroc1/k, v_auroc2/k)}
            
    for key in errd.keys():
        acc = []
        for i in range(len(errd[key])):
            acc.append(vrrd[key][i]-errd[key][i]**2)
        errd[key + '_v'] = acc
            

    params_str =  solver + '\t' + 'lamP=%f lamR=%f lamS=%f' % (lamP, lamR, lamS)
    
    result_str_l = []
    header_str_l = ['params']
    for k in ['mse','R2','aupr','auroc']:
        kv = k+'_v'
        result_str_l.append(errd[k][0])
        result_str_l.append(errd[kv][0])
        result_str_l.append(errd[k][1])
        result_str_l.append(errd[kv][1])
        header_str_l.append(k+'1')
        header_str_l.append(kv+'1')
        header_str_l.append(k+'2')
        header_str_l.append(kv+'2')
    result_str = '\t'.join(map(str, result_str_l))
    header_str = '\t'.join(map(str, header_str_l))                       
    
    print params_str + '\t' + result_str + '\n'
    print 'correlation of fused coefficients is %f' % corr
    if not os.path.exists(os.path.join(data_fn, 'results')):
        with open(os.path.join(data_fn, 'results'),'w') as outf:
            outf.write(header_str + '\n')
    with open(os.path.join(data_fn, 'results'),'a') as outf:
        outf.write(params_str + '\t' + result_str + '\n')
    with open(os.path.join(data_fn, 'last_results'),'a') as outf:
        outf.write(header_str + '\n')
        outf.write(params_str + '\t' + result_str + '\n')

    
    return errd


#SECTION: -------------------------CODE FOR EVALUATING THE OUTPUT

#model prediction error, using one of several metrics
#exclude_tfs doesn't evaluate predictions of the tfs, and assumes that TFs come before all other genes.
def prediction_error(X, B, Y, metric, exclude_tfs = True):
    Ypred = np.dot(X, B)
    y = Y[:,0]
    yp = Ypred[:,0]
    num_tfs = B.shape[0]
    if exclude_tfs:
        start_ind = num_tfs
    else:
        start_ind = 0
    
    if metric == 'R2':
        r2a = 0.0
        from matplotlib import pyplot as plt
        for c in range(start_ind, Ypred.shape[1]):
            y = Y[:, c]
            yp = Ypred[:, c]
            r2 = 1 - ((y-yp)**2).sum()/ ((y-y.mean())**2).sum()
            r2a += r2
            
#            if c == start_ind:
#                plt.plot(y)
#                plt.plot(yp)
#                plt.show()
        return r2a/(Ypred.shape[1]-start_ind)
    if metric == 'mse':
        msea = 0.0
        for c in range(start_ind, Ypred.shape[1]):
            y = Y[:, c]
            yp = Ypred[:, c]
            mse = ((y-yp)**2).mean()
            msea += mse

        return msea / (Ypred.shape[1]-start_ind)
    if metric == 'corr':
        corra = 0.0
        for c in range(start_ind, Ypred.shape[1]):
            y = Y[:, c]
            yp = Ypred[:, c]
            corr = np.corrcoef(y, yp)[0,1]
            corra += corr
        return corra / (Ypred.shape[1] - start_ind)

#evaluates the area under the precision recall curve, with respect to some given priors
# exclude_tfs: do not evaluate on tf x tf interactions
# constraints: if not None, evaluates only on interactions which have fusion constraints
#sub: name of subproblem. used if constraints != None
def eval_network_pr(net, genes, tfs, priors, exclude_tfs = False, constraints = None, sub=None):
    #from matplotlib import pyplot as plt
    org = priors[0][0].organism
    priors_set = set(priors)
    gene_to_ind = {genes[x] : x for x in range(len(genes))}
    
    tf_to_ind = {tfs[x] : x for x in range(len(tfs))}
    gene_marked = np.zeros(len(genes)) != 0
    tf_marked = np.zeros(len(tfs)) != 0
    if constraints != None:
        con_set = set()
        for con in constraints:    
            if con.c1.sub == sub:
                con_set.add(con.c1)
            if con.c2.sub == sub:
                con_set.add(con.c2)
        
    #we only evaluate on interactions when the gene/tf is mentioned in a prior
    for prior in priors:
        
        gene_marked[gene_to_ind[prior[0].name]] = True
        gene_marked[gene_to_ind[prior[1].name]] = True
        if prior[0].name in tf_to_ind:
            tf_marked[tf_to_ind[prior[0].name]] = True
        if prior[1].name in tf_to_ind:
            tf_marked[tf_to_ind[prior[1].name]] = True

    genes = np.array(genes)[gene_marked]
    tfs = np.array(tfs)[tf_marked]
    net = net[:, gene_marked]
    net = net[tf_marked, :]
    scores = []#np.zeros(len(genes)*len(tfs))
    labels = []#np.zeros(len(genes)*len(tfs))
    i=0
    for tfi in range(len(tfs)):
        for gi in range(len(genes)):
            if exclude_tfs and gi < len(tfs):
                continue
            if constraints != None:
                coeff = fl.coefficient(sub=sub, r=tfi, c=gi) #potential coefficient
                
                if not coeff in con_set:
                    continue

            tf = tfs[tfi]
            g = genes[gi]
            score = np.abs(net[tfi, gi])
            label = 0
            if (fl.one_gene(tf, org), fl.one_gene(g, org)) in priors_set:
                label = 1
            if (fl.one_gene(g, org), fl.one_gene(tf, org)) in priors_set:
                label = 1
            
            
            scores.append(score)#scores[i] = score
            labels.append(label)#labels[i] = label
            
    
    
    (precision, recall,t) = precision_recall_curve(labels, scores)#prc(scores, labels)
    aupr = auc(recall, precision)
    
    return aupr

#evaluates the area under the roc, with respect to some given priors

def eval_network_roc(net, genes, tfs, priors, exclude_tfs = True, constraints = None, sub=None):
    
    org = priors[0][0].organism
    priors_set = set(priors)
    scores = []#np.zeros(len(genes)*len(tfs))
    labels = []#np.zeros(len(genes)*len(tfs))
    i=0
    if constraints != None:
        con_set = set()
        for con in constraints:
            if con.c1.sub == sub:
                con_set.add(con.c1)
            if con.c2.sub == sub:
                con_set.add(con.c2)
        
    for tfi in range(len(tfs)):
        for gi in range(len(genes)):
            if exclude_tfs and gi < len(tfs):
                continue
            if constraints != None:
                coeff = fl.coefficient(sub=sub, r=tfi, c=gi) #potential coefficient

                if not coeff in con_set:
                    continue

            tf = tfs[tfi]
            g = genes[gi]
            score = np.abs(net[tfi, gi])
            label = 0
            
            
            if (fl.one_gene(tf, org), fl.one_gene(g, org)) in priors_set:
                label = 1
            if (fl.one_gene(g, org), fl.one_gene(tf, org)) in priors_set:
                label = 1
            scores.append(score)
            labels.append(label)
            
            #scores[i] = score
            #labels[i] = label
            i += 1
    (fpr, tpr, t) = roc_curve(labels, scores)
    if any(np.isnan(fpr)) or any(np.isnan(tpr)):
        return 0.0 #no false positives
    
    auroc = auc(fpr, tpr)
    return auroc

def eval_network_beta(net1, net2):
    return ((net1 - net2)**2).mean()
            
#generates fusion constraints, then computes the correlation between fused coefficients
def fused_coeff_corr(organisms, genes_l, tfs_l, orth, B_l):
    constraints = fl.orth_to_constraints(organisms, genes_l, tfs_l, orth, 1.0)
    fused_vals = [[],[]]
    print 'there are %d constraints'%len(constraints)
    if len(constraints) == 0:
        return (np.nan, np.zeros((2,0)))
    for con in constraints:
        s1 = con.c1.sub
        b1 = B_l[s1][con.c1.r, con.c1.c]
        s2 = con.c2.sub
        b2 = B_l[s2][con.c2.r, con.c2.c]
        fused_vals[s1].append(b1)
        fused_vals[s2].append(b2)
    fused_vals = np.array(fused_vals)
    return (np.corrcoef(fused_vals)[0,1], fused_vals)

#take list of lamP, lamR, lamS values and finds the optimal parameters using cv_model1
def grid_search_params(data_fn, lamP, lamR, lamS, k, solver='solve_ortho_direct',special_args=None, reverse=False, cv_both=(True,True), exclude_tfs=True, eval_metric='mse'):
    grid = dict()
    best_mse = 1000
    best_R2 = 0
    best_aupr = 0
    best_auroc = 0
    best_lamP = 1.0
    best_lamR = 0
    best_lamS = 0
    for r in range(len(lamR)):
        for s in range(len(lamS)):
            for p in range(len(lamP)):
                errd = cv_model1(data_fn, lamP[p], lamR[r], lamS[s], k, solver='solve_ortho_direct',special_args=None, reverse=False, cv_both=(True,True), exclude_tfs=True)
                if eval_metric == 'mse':
                    grid[str(lamR[r])+'_'+str(lamS[s])+'_'+str(lamP[p])] = errd['mse']
                    score = 0.5*(errd['mse'][0]+errd['mse'][1])
                    if score < best_mse:
                        best_mse = score
                        best_lamP = lamP[p]
                        best_lamR = lamR[r]
                        best_lamS = lamS[s]
                if eval_metric == 'R2':
                    grid[str(lamR[r])+'_'+str(lamS[s])+'_'+str(lamP[p])] = errd['R2']   
                    score = 0.5*(errd['R2'][0]+errd['R2'][1])
                    if score > best_R2:
                        best_R2 = score
                        best_lamP = lamP[p]
                        best_lamR = lamR[r]
                        best_lamS = lamS[s]
                if eval_metric == 'aupr':
                    grid[str(lamR[r])+'_'+str(lamS[s])+'_'+str(lamP[p])] = errd['aupr']
                    score = 0.5*(errd['aupr'][0]+errd['aupr'][1])
                    if score > best_aupr:
                        best_aupr = score
                        best_lamP = lamP[p]
                        best_lamR = lamR[r]
                        best_lamS = lamS[s]
                if eval_metric == 'auroc':
                    grid[str(lamR[r])+'_'+str(lamS[s])+'_'+str(lamP[p])] = errd['auroc']
                    score = 0.5*(errd['auroc'][0]+errd['auroc'][1])
                    if score > best_auroc:
                        best_auroc = score
                        best_lamP = lamP[p]
                        best_lamR = lamR[r]
                        best_lamS = lamS[s]
    if eval_metric == 'mse':
        return (best_mse, best_lamP, best_lamR, best_lamS, grid)
    if eval_metric == 'R2':
        return (best_R2, best_lamP, best_lamR, best_lamS, grid)
    if eval_metric == 'aupr':
        return (best_aupr, best_lamP, best_lamR, best_lamS, grid)
    if eval_metric == 'auroc':
        return (best_auroc, best_lamP, best_lamR, best_lamS, grid)

