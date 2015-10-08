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
    lamP = 0.001
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
    use_TFA=False

    for i in range(it):
        seed=i*abs(np.random.randn())
        lamS=0
        uf = fg.cv_model_m(net_path,lamP,lamR,lamS,k,solver,settings_fs,reverse,cv_both,exclude_tfs,1,seed,verbose,orth_file,orgs,lamS_opt,test_all, use_TFA)
        for i in range(k):
            scores['solver'].append('unfused')
            scores['mse 1'].append(uf[0]['mse'][i][0])
            scores['mse 2'].append(uf[1]['mse'][i][0])
            scores['aupr 1'].append(uf[0]['aupr'][i][0])
            scores['aupr 2'].append(uf[1]['aupr'][i][0])

        solver='solve_ortho_direct'
        lamS = 0.5
        orth_file=['orth']
        l2 = fg.cv_model_m(net_path,lamP,lamR,lamS,k,solver,settings_fs,reverse,cv_both,exclude_tfs,1,seed,verbose,orth_file,orgs,lamS_opt, test_all, use_TFA)
        for i in range(k):
            scores['solver'].append('L2')
            scores['mse 1'].append(l2[0]['mse'][i][0])
            scores['mse 2'].append(l2[1]['mse'][i][0])
            scores['aupr 1'].append(l2[0]['aupr'][i][0])
            scores['aupr 2'].append(l2[1]['aupr'][i][0])   

        solver='solve_ortho_direct'
        lamS = 2
        orth_file=['orth']
        l2_x = fg.cv_model_m(net_path,lamP,lamR,lamS,k,solver,settings_fs,reverse,cv_both,exclude_tfs,1,seed,verbose,orth_file,orgs,lamS_opt, test_all,use_TFA)
        for i in range(k):
            scores['solver'].append('L2.5')
            scores['mse 1'].append(l2_x[0]['mse'][i][0])
            scores['mse 2'].append(l2_x[1]['mse'][i][0])
            scores['aupr 1'].append(l2_x[0]['aupr'][i][0])
            scores['aupr 2'].append(l2_x[1]['aupr'][i][0])   

    return (scores, uf, l2, l2_x)

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
        
    prior_cons = fr.priors_to_constraints([ds1.name], [genes1], [tfs1], priors1, [1.0])
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


#looks at (tf, g) correlations and takes the top k (tf, g) pairs and looks at their priors
def compare_expcorr_prior(k):
    dfk = pd.DataFrame.from_csv('data/TH17/th17_whole_K_cut_prcnt_20_num_tfs_28_sam_0_deseq_cut_0.25_Aug_8_2012_priorCut0p75.tsv',sep='\t',header=0)
    dfc = pd.DataFrame.from_csv('data/TH17/th17_whole_C_cut_prcnt_0_num_tfs_28_sam_0_deseq_cut_1_Aug_8_2012_priorCut0p75.tsv',sep='\t',header=0)

    bactf = 'data/Th17_standard'
    ds1 = ds.standard_source(bactf,0)
    ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_gold()
    
    (priors2, signs2) = ds2.get_priors()
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
    
    inds_gene_ma = {i : genes1[i] for i in range(len(genes1))}
    inds_tf_ma = {i : tfs1[i] for i in range(len(tfs1))}
    inds_gene_r = {i : genes2[i] for i in range(len(genes2))}
    inds_tf_r = {i : tfs2[i] for i in range(len(tfs2))}

    prior_genes = list(dfk.index)
    prior_tfs = list(dfk.columns)

    prior_interactions_ma = np.zeros((len(tfs1),len(genes1)))
    top_prior_inds_ma = []
    prior_int_tfa_ma = np.zeros((len(tfs1),len(genes1)))
    top_prior_inds_tfa_ma = []

    for i in range(len(tfs1)):
        for j in range(len(genes1)):
            exp_tf = e1_tr[:,i]
            exp_g = e1_tr[:,j]
            corr = np.corrcoef(exp_tf,exp_g)[1,0]
            prior_interactions_ma[i,j] = corr
    prior_interactions_ma_f = prior_interactions_ma.flatten()
    top_prior_inds_ma_f = prior_interactions_ma_f.argsort()[len(tfs1):k+len(tfs1)][::-1]

    for i in range(k):
        indf = top_prior_inds_ma_f[i]
        gind = np.mod(indf, len(genes1))
        tind = indf/len(genes1)
        top_prior_inds_ma.append((tind, gind))

    prior_vals_ma = []
    rand_vals_ma = []
    for i in range(k):
        (t, g) = top_prior_inds_ma[i]
        tf = inds_tf_ma[t]
        gene = inds_gene_ma[g]
        if tf in prior_tfs:
            if gene in prior_genes:
                prior_vals_ma.append(dfk[tf][gene])
    prior_vals_ma = np.array(prior_vals_ma)

    count = 0
    for i in range(len(prior_interactions_ma_f)):
        t = np.random.randint(0, len(tfs1))
        g = np.random.randint(0, len(genes1))
        if count < len(prior_vals_ma):
            if tfs1[t] in prior_tfs:
                if genes1[g] in prior_genes:
                    rand_vals_ma.append(dfk[tfs1[t]][genes1[g]])
                    count+=1
    rand_vals_ma = np.array(rand_vals_ma)



    for i in range(len(tfs1)):
        for j in range(len(tfs1)):
            exp_t1 = t1_tr[:,i]
            exp_t2 = t1_tr[:,j]
            corr = np.corrcoef(exp_t1,exp_t2)[1,0]
            prior_int_tfa_ma[i,j] = corr
    prior_int_tfa_ma_f = prior_int_tfa_ma.flatten()
    top_prior_inds_tfa_ma_f = prior_int_tfa_ma_f.argsort()[len(tfs1):k+len(tfs1)][::-1]

    for i in range(k):
        indf = top_prior_inds_tfa_ma_f[i]
        tind2 = np.mod(indf, len(tfs1))
        tind1 = indf/len(genes1)
        top_prior_inds_tfa_ma.append((tind1, tind2))

    prior_vals_tfa_ma = []
    rand_vals_tfa_ma = []
    for i in range(k):
        (t1, t2) = top_prior_inds_tfa_ma[i]
        tf1 = inds_tf_ma[t1]
        tf2 = inds_tf_ma[t2]
        if tf1 in prior_tfs:
            if tf2 in prior_tfs:
                prior_vals_tfa_ma.append(dfk[tf1][tf2])
    prior_vals_tfa_ma = np.array(prior_vals_tfa_ma)

    count = 0
    for i in range(len(prior_int_tfa_ma_f)):
        t1 = np.random.randint(0, len(tfs1))
        t2 = np.random.randint(0, len(tfs1))
        if count < len(prior_vals_tfa_ma):
            if tfs1[t1] in prior_tfs:
                if tfs1[t2] in prior_genes:
                    rand_vals_tfa_ma.append(dfk[tfs1[t1]][tfs1[t2]])
                    count+=1
    rand_vals_tfa_ma = np.array(rand_vals_tfa_ma)



    prior_interactions_r = np.zeros((len(tfs2),len(genes2)))
    top_prior_inds_r = []

    for i in range(len(tfs2)):
        for j in range(len(genes2)):
            exp_tf = e2_tr[:,i]
            exp_g = e2_tr[:,j]
            corr = np.corrcoef(exp_tf,exp_g)[1,0]
            prior_interactions_r[i,j] = corr
    prior_interactions_r_f = prior_interactions_r.flatten()
    top_prior_inds_r_f = prior_interactions_r_f.argsort()[len(tfs2):k+len(tfs2)][::-1]

    for i in range(k):
        indf = top_prior_inds_r_f[i]
        gind = np.mod(indf, len(genes2))
        tind = indf/len(genes2)
        top_prior_inds_r.append((tind, gind))

    prior_vals_r = []
    rand_vals_r = []
    for i in range(k):
        (t, g) = top_prior_inds_r[i]
        tf = inds_tf_r[t]
        gene = inds_gene_r[g]
        if tf in prior_tfs:
            if gene in prior_genes:
                prior_vals_r.append(dfk[tf][gene])
    prior_vals_r = np.array(prior_vals_r)

    count = 0
    for i in range(len(prior_interactions_r_f)):
        t = np.random.randint(0, len(tfs2))
        g = np.random.randint(0, len(genes2))
        if count < len(prior_vals_r):
            if tfs2[t] in prior_tfs:
                if genes2[g] in prior_genes:
                    rand_vals_r.append(dfk[tfs2[t]][genes2[g]])
                    count+=1
    rand_vals_r = np.array(rand_vals_r)

    sns.kdeplot(prior_vals_tfa_ma, shade=True, label='priors')
    plt.hold(True)
    sns.kdeplot(rand_vals_tfa_ma, shade=True, label='non-priors')
    plt.xlabel('KO score')
    plt.legend()
    plt.ylabel('frequency')
    plt.title('microarray TFA')
    print 'prior interactions %f, random interactions %f' %(np.mean(abs(prior_vals_ma)), np.mean(abs(rand_vals_ma)))
    plt.show()

    sns.kdeplot(prior_vals_ma, shade=True, label='priors')
    plt.hold(True)
    sns.kdeplot(rand_vals_ma, shade=True, label='non-priors')
    plt.xlabel('KO score')
    plt.legend()
    plt.ylabel('frequency')
    plt.title('microarray')
    print 'prior interactions %f, random interactions %f' %(np.mean(abs(prior_vals_ma)), np.mean(abs(rand_vals_ma)))
    plt.show()


    sns.kdeplot(prior_vals_r, shade=True, label='highest correlated (TF, gene) pairs')
    plt.hold(True)
    sns.kdeplot(rand_vals_r, shade=True, label='random (TF, gene) pairs')
    plt.xlabel('KO score')
    plt.legend()
    plt.ylabel('frequency')
    plt.title('RNA seq')
    print 'prior interactions %f, random interactions %f' %(np.mean(abs(prior_vals_r)), np.mean(abs(rand_vals_r)))
    plt.show()
