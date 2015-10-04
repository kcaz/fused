import os
import data_sources as ds
import fused_reg as fr
import fit_grn as fg
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def rna_ma_test():
    it = 1
    scores = {'solver':[], 'mse 1':[], 'mse 2':[], 'aupr 1':[], 'aupr 2':[]}

    net_path = 'data/Th17_standard'
    k = 1
    lamP = 0.5
    lamR= 3
    orth_file=['orth']
    orgs=['microarray','RNAseq']
    reverse=False
    cv_both=(True,True)
    exclude_tfs=True
    seed=np.random.randn()
    verbose=False
    lamS_opt=None
    solver='solve_ortho_direct'
    test_all='all'
    settings_fs=fr.get_settings()
    use_TFA=True

    for i in range(it):
        seed=i*abs(np.random.randn())
        lamS=0
        uf = fg.cv_model_m(net_path,lamP,lamR,lamS,k,solver,settings_fs,reverse,cv_both,exclude_tfs,100,seed,verbose,orth_file,orgs,lamS_opt,test_all, use_TFA)
        for i in range(k):
            scores['solver'].append('unfused')
            scores['mse 1'].append(uf[0]['mse'][i][0])
            scores['mse 2'].append(uf[1]['mse'][i][0])
            scores['aupr 1'].append(uf[0]['aupr'][i][0])
            scores['aupr 2'].append(uf[1]['aupr'][i][0])

        solver='solve_ortho_direct'
        lamS = 0.5
        orth_file=['orth']
        l2_x = fg.cv_model_m(net_path,lamP,lamR,lamS,k,solver,settings_fs,reverse,cv_both,exclude_tfs,100,seed,verbose,orth_file,orgs,lamS_opt, test_all, use_TFA)
        for i in range(k):
            scores['solver'].append('L2')
            scores['mse 1'].append(l2_x[0]['mse'][i][0])
            scores['mse 2'].append(l2_x[1]['mse'][i][0])
            scores['aupr 1'].append(l2_x[0]['aupr'][i][0])
            scores['aupr 2'].append(l2_x[1]['aupr'][i][0])   

        solver='solve_ortho_direct'
        lamS = 2
        orth_file=['orth']
        l2_x = fg.cv_model_m(net_path,lamP,lamR,lamS,k,solver,settings_fs,reverse,cv_both,exclude_tfs,100,seed,verbose,orth_file,orgs,lamS_opt, test_all,use_TFA)
        for i in range(k):
            scores['solver'].append('L2.5')
            scores['mse 1'].append(l2_x[0]['mse'][i][0])
            scores['mse 2'].append(l2_x[1]['mse'][i][0])
            scores['aupr 1'].append(l2_x[0]['aupr'][i][0])
            scores['aupr 2'].append(l2_x[1]['aupr'][i][0])   

    return scores


def test_prior_sign_corr():
    print 'hallo'
    bactf = 'data/bacteria_standard'
    ds1 = ds.standard_source(bactf,0)
    #ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_priors()
    
    #(priors2, signs2) = ds2.get_priors()
    #(constraints, marks, orths) = ds.load_constraints(bactf)
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    #(e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
    
    priors_a = map(lambda ind1: priors1[ind1], filter(lambda ind2: signs1[ind2] == 1, range(len(signs1))))

    priors_r = map(lambda ind1: priors1[ind1], filter(lambda ind2: signs1[ind2] == -1, range(len(signs1))))

    prior_cons_r = fr.priors_to_constraints([ds1.name], [genes1], [tfs1], priors_r, 1.0)
    prior_cons_a = fr.priors_to_constraints([ds1.name], [genes1], [tfs1], priors_a, 1.0)
    #gene_corr = np.corrcoef(e1_tr.T)
    #we now rely on the fact that tfs occur before genes, and in the same order as they do in the list of genes, so indices are the same
    
    repr_interactions = np.zeros(len(prior_cons_r))
    acti_interactions = np.zeros(len(prior_cons_a))
    rand_interactions = np.zeros(len(prior_cons_a))
    for i, prior in enumerate(prior_cons_r):
        tfi = prior.c1.r
        gi = prior.c1.c
        exp_tf = e1_tr[:,[tfi]]
        exp_g = e1_tr[:, [gi]]
        
        corrmat = np.corrcoef(np.hstack((exp_tf, exp_g)).T)
        
        corr = (corrmat[1,0])
        repr_interactions[i] = corr
    for i, prior in enumerate(prior_cons_a):
        tfi = prior.c1.r
        gi = prior.c1.c
        exp_tf = e1_tr[:,[tfi]]
        exp_g = e1_tr[:, [gi]]
        corrmat = np.corrcoef(np.hstack((exp_tf, exp_g)).T)        
        corr = (corrmat[1,0])
        acti_interactions[i] = corr

    for i, prior in enumerate(prior_cons_a):
        tfi = np.random.randint(0, len(tfs1))
        gi = np.random.randint(0, len(genes1))
        exp_tf = e1_tr[:,[tfi]]
        exp_g = e1_tr[:, [gi]]
        
        corrmat = np.corrcoef(np.hstack((exp_tf, exp_g)).T)
        corr = (corrmat[1,0])
        rand_interactions[i] = corr


    sns.kdeplot(repr_interactions, shade=True, label = 'repression, %f'% np.mean(repr_interactions))
    plt.hold(True)
    sns.kdeplot(acti_interactions, shade=True, label = 'activation, %f'% np.mean(acti_interactions))
    sns.kdeplot(rand_interactions, shade=True, label = 'non-priors, %f' % np.mean(rand_interactions))
    
    
    plt.xlabel('correlation')
    plt.legend()
    plt.ylabel('frequency')    
    
    plt.show()



# we look at the TF x Gene correlation matrices in each species, for genes which have orthologies. I'll then order one of them, force the same order on the other species, take the difference of correlations, and compare with difference of correlation of random pairs. modified from experiments.py
def check_ortho_corr():
    bactf = 'data/Th17_standard'
    ds1 = ds.standard_source(bactf,0)
    ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_priors()
    
    (priors2, signs2) = ds2.get_priors()
    (constraints, marks, orths) = ds.load_constraints(bactf)
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
    
    tfs_set_subt = set(tfs1)
    tfs_set_anth = set(tfs2)

    orth_set_subt = set(map(lambda orth: orth.genes[0].name, orths))
    orth_set_anth = set(map(lambda orth: orth.genes[1].name, orths))
    #map from one to the other, OK if orthology 1-1
    subt_to_anth = {orths[x].genes[0].name : orths[x].genes[1].name for x in range(len(orths))}
    anth_to_subt = {orths[x].genes[1].name : orths[x].genes[0].name for x in range(len(orths))}


    tfs_orth_subt = filter(lambda tf: tf in orth_set_subt and subt_to_anth[tf] in tfs_set_anth, tfs1)
    tfs_orth_anth = filter(lambda tf: tf in orth_set_anth and anth_to_subt[tf] in tfs_set_subt, tfs2)

    gen_orth_subt = filter(lambda g: g in orth_set_subt, genes1)
    gen_orth_anth = filter(lambda g: g in orth_set_anth, genes2)

    gene_inds_subt = {genes1[i] : i for i in range(len(genes1))}
    gene_inds_anth = {genes2[i] : i for i in range(len(genes2))}

    subt_corr_mat = np.zeros((len(tfs_orth_subt), len(gen_orth_subt)))
    anth_corr_mat = np.zeros((len(tfs_orth_anth), len(gen_orth_anth)))
   #subt_corr_mat = np.random.randn(50,50)#len(tfs_orth_subt), len(gen_orth_subt))
    #anth_corr_mat = np.random.randn(50,50)#len(tfs_orth_anth), len(gen_orth_anth))
    
    
    for r, tf in enumerate(tfs_orth_subt):
        for c, g in enumerate(gen_orth_subt):
            
            tfi = gene_inds_subt[tf]
            gi = gene_inds_subt[g]
            corr = np.corrcoef( np.hstack((e1_tr[:, [tfi]], e1_tr[:, [gi]])).T )[0,1]
            
            
            subt_corr_mat[r, c] = corr
    anth_to_subt = {orths[x].genes[1].name : orths[x].genes[0].name for x in range(len(orths))}
    #for the anthracis correlations, we map the tf or gene onto the corresponding subtilis gene, then compute that index
    tfs_orth_subt_ind = {tfs_orth_subt[i] : i for i in range(len(tfs_orth_subt))}
    gen_orth_subt_ind = {gen_orth_subt[i] : i for i in range(len(gen_orth_subt))}
    for r, tf in enumerate(tfs_orth_anth):
        for c, g in enumerate(gen_orth_anth):
            tfi = gene_inds_anth[tf]
            gi = gene_inds_anth[g]
            corr = np.corrcoef( np.hstack((e2_tr[:, [tfi]], e2_tr[:, [gi]])).T )[0, 1]

            tf_subt = anth_to_subt[tf]
            g_subt = anth_to_subt[g]
       
            r_subt = tfs_orth_subt_ind[tf_subt]
            c_subt = gen_orth_subt_ind[g_subt]
            
            anth_corr_mat[r_subt, c_subt] = corr    
            
    subt_corrs = subt_corr_mat.ravel()
    anth_corrs = anth_corr_mat.ravel()
    
    sns.kdeplot(np.abs(subt_corrs - anth_corrs), shade=True, label = 'orthologs, %f'% np.mean(np.abs(subt_corrs - anth_corrs)))
    plt.hold(True)
    np.random.shuffle(subt_corrs)
    sns.kdeplot(np.abs(subt_corrs - anth_corrs), shade=True, label = 'non-orths, %f'% np.mean(np.abs(subt_corrs - anth_corrs)))
    plt.xlabel('correlation difference')
    plt.ylabel('frequency')
    plt.show()


#looks at the distribution of absolute values of correlations for prior interactions, and an equally sized set of random interactions
def test_prior_corr():
    
    bactf = 'data/Th17_standard'
    ds1 = ds.standard_source(bactf,0)
    #ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_gold()
    
    #(priors2, signs2) = ds2.get_priors()
    #(constraints, marks, orths) = ds.load_constraints(bactf)
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    #(e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
    
    prior_cons = fr.priors_to_constraints([ds1.name], [genes1], [tfs1], priors1, 1.0)
    #gene_corr = np.corrcoef(e1_tr.T)
    #we now rely on the fact that tfs occur before genes, and in the same order as they do in the list of genes, so indices are the same
    
    prior_interactions = np.zeros(len(prior_cons))
    random_interactions = np.zeros(len(prior_cons))
    for i, prior in enumerate(prior_cons):
        tfi = prior.c1.r
        gi = prior.c1.c
        exp_tf = e1_tr[:,[tfi]]
        exp_g = e1_tr[:, [gi]]
        
        corrmat = np.corrcoef(np.hstack((exp_tf, exp_g)).T)
        
        corr = (corrmat[1,0])
        prior_interactions[i] = corr

    for i, prior in enumerate(prior_cons):
        tfi = np.random.randint(0, len(tfs1))
        gi = np.random.randint(0, len(genes1))
        exp_tf = e1_tr[:,[tfi]]
        exp_g = e1_tr[:, [gi]]
        
        corrmat = np.corrcoef(np.hstack((exp_tf, exp_g)).T)
        
        corr = (corrmat[1,0])
        random_interactions[i] = corr

    
    

    sns.kdeplot(prior_interactions, shade=True, label='priors')
    
    plt.hold(True)
    sns.kdeplot(random_interactions, shade=True, label='non-priors')
    plt.xlabel('correlation')
    plt.legend()
    plt.ylabel('frequency')
    print 'prior interactions %f, random interactions %f' %(np.mean(prior_interactions), np.mean(random_interactions))
    plt.show()