import numpy as np
import fused_reg as fl
import data_sources as ds
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import collections
#SECTION: ------------------UTILITY FUNCTIONS-----------------




#SECTION: ------------------FOR RUNNING BACTERIAL DATA

#this is basic - loads all the data, fits the basic model, and returns B
def test_bacteria(lamP, lamR, lamS):
    subt = ds.subt()
    anthr = ds.anthr()
    
    (bs_e, bs_t, bs_genes, bs_tfs) = subt.load_data()
    (ba_e, ba_t, ba_genes, ba_tfs) = anthr.load_data()

    (bs_priors, bs_sign) = subt.get_priors()
    (ba_priors, ba_sign) = anthr.get_priors()


    Xs = [bs_t, ba_t]
    Ys = [bs_e, ba_e]
    genes = [bs_genes, ba_genes]
    tfs = [bs_tfs, ba_tfs]
    priors = bs_priors + ba_priors
    orth = ds.ba_bs_orth()
    organisms = [subt.name, anthr.name]

    Bs = fl.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)

    return Bs


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
    
    return Bs

#runs the basic model with specified parameters under k-fold cross-validation, and stores a number of metrics
#k: the number of cv folds
#reverse: train on the little dude (reverse train and test)
def cv_model1(data_fn, lamP, lamR, lamS, k, solver='solve_ortho_direct',special_args=None, reverse=False, cv_both=(True,True)):
    ds1 = ds.standard_source(data_fn,0)
    ds2 = ds.standard_source(data_fn,1)
    orth_fn = os.path.join(data_fn, 'orth')

    organisms = [ds1.name, ds2.name]
    orth = ds.load_orth(orth_fn, organisms)
    (mse1, mse2, R21, R22, aupr1, aupr2, auroc1, auroc2) = list(np.zeros(8))

    
    folds1 = ds1.partition_data(k)
    folds2 = ds2.partition_data(k)

    excl = lambda x,i: x[0:i]+x[(i+1):]
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()
    #initialize a bunch of things to accumulate over
    

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
            Bs = fl.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, s_it = special_args['s_it'])
        

        mse1 += prediction_error(t1_te, Bs[0], e1_te, 'mse')
        mse2 += prediction_error(t2_te, Bs[1], e2_te, 'mse')

        R21 += prediction_error(t1_te, Bs[0], e1_te, 'R2')
        R22 += prediction_error(t2_te, Bs[1], e2_te, 'R2')

        if len(priors1):
            aupr1 += eval_network_pr(Bs[0], genes1, tfs1, priors1)
        if len(priors2):
            aupr2 += eval_network_pr(Bs[1], genes2, tfs2, priors2)

        if len(priors1):
            auroc1 += eval_network_roc(Bs[0], genes1, tfs1, priors1)
        if len(priors2):        
            auroc2 += eval_network_roc(Bs[1], genes2, tfs2, priors2)

    params_str = 'simple lamP=%f lamR=%f lamS=%f' % (lamP, lamR, lamS)
    
    result_str = '\t'.join(8*['%f']) % (mse1/k, mse2/k, R21/k, R22/k, aupr1/k, aupr2/k, auroc1/k, auroc2/k)
    print params_str + '\t' + result_str + '\n'

    if not os.path.exists(os.path.join(data_fn, 'results')):
        with open(os.path.join(data_fn, 'results'),'w') as outf:
            outf.write('params\tmse1\tmse2\tR21\tR22\taupr1\taupr2\tauroc1\tauroc2\n')
    with open(os.path.join(data_fn, 'results'),'a') as outf:
        outf.write(params_str + '\t' + result_str + '\n')
    with open(os.path.join(data_fn, 'last_results'),'a') as outf:
        outf.write('params\tmse1\tmse2\tR21\tR22\taupr1\taupr2\tauroc1\tauroc2\n')
        outf.write(params_str + '\t' + result_str + '\n')

    ret_dict = {'mse':(mse1/k, mse2/k), 'R2':(R21/k, R22/k), 'aupr':(aupr1/k, aupr2/k),'auroc':(auroc1/k, auroc2/k)}
    return ret_dict
#SECTION: -------------------------CODE FOR EVALUATING THE OUTPUT

#model prediction error, using one of several metrics.
def prediction_error(X, B, Y, metric):
    Ypred = np.dot(X, B)
    y = Y[:,0]
    yp = Ypred[:,0]

    if metric == 'R2':
        r2a = 0.0
        for c in range(Ypred.shape[1]):
            y = Y[:, c]
            yp = Ypred[:, c]
            r2 = 1 - ((y-yp)**2).sum()/ ((y-y.mean())**2).sum()
            r2a += r2
            
        return r2a/Ypred.shape[1]
    if metric == 'mse':
        msea = 0.0
        for c in range(Ypred.shape[1]):
            y = Y[:, c]
            yp = Ypred[:, c]
            mse = ((y-yp)**2).mean()
            msea += mse

        return msea / Ypred.shape[1]
    if metric == 'corr':
        corra = 0.0
        for c in range(Ypred.shape[1]):
            y = Y[:, c]
            yp = Ypred[:, c]
            corr = np.corrcoef(y, yp)[0,1]
            corra += corr
        return corra / Ypred.shape[1]

#evaluates the area under the precision recall curve, with respect to some given priors
def eval_network_pr(net, genes, tfs, priors):
    #from matplotlib import pyplot as plt
    org = priors[0][0].organism
    priors_set = set(priors)
    gene_to_ind = {genes[x] : x for x in range(len(genes))}
    tf_to_ind = {tfs[x] : x for x in range(len(tfs))}
    gene_marked = np.zeros(len(genes)) != 0
    tf_marked = np.zeros(len(tfs)) != 0
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
    scores = np.zeros(len(genes)*len(tfs))
    labels = np.zeros(len(genes)*len(tfs))
    i=0
    for tfi in range(len(tfs)):
        for gi in range(len(genes)):
            tf = tfs[tfi]
            g = genes[gi]
            score = np.abs(net[tfi, gi])
            label = 0
            if (fl.one_gene(tf, org), fl.one_gene(g, org)) in priors_set:
                label = 1
            if (fl.one_gene(g, org), fl.one_gene(tf, org)) in priors_set:
                label = 1
            scores[i] = score
            labels[i] = label
            i += 1
    (precision, recall,t) = precision_recall_curve(labels, scores)#prc(scores, labels)
    aupr = auc(recall, precision)
    
    return aupr

#evaluates the area under the roc, with respect to some given priors
def eval_network_roc(net, genes, tfs, priors):
    #from matplotlib import pyplot as plt
    org = priors[0][0].organism
    priors_set = set(priors)
    scores = np.zeros(len(genes)*len(tfs))
    labels = np.zeros(len(genes)*len(tfs))
    i=0
    for tfi in range(len(tfs)):
        for gi in range(len(genes)):
            tf = tfs[tfi]
            g = genes[gi]
            score = np.abs(net[tfi, gi])
            label = 0
            if (fl.one_gene(tf, org), fl.one_gene(g, org)) in priors_set:
                label = 1
            if (fl.one_gene(g, org), fl.one_gene(tf, org)) in priors_set:
                label = 1
            scores[i] = score
            labels[i] = label
            i += 1
    (fpr, tpr, t) = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    #    plt.plot(fpr, tpr)
    #   plt.show()

    return auroc


            
