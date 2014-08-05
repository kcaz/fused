import numpy as np
import fused_L2 as fl

iron_conds = 24
timeseries_conds = 51
subtilis_conds = 359

def run_scr(lamP, lamR, lamS, outf, sub):
    #(ba,tf, genes, tfs) = load_bacteria('B_subtilis.csv','tfNames.txt',range(250))
    (ba,tf, genes, tfs) = load_B_subtilis(sub)
    (BC_priors, sign) = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    orth = load_orth('',['B_subtilis'])

    print ba.shape
    organisms = ['B_subtilis']
    gene_ls = [genes]
    tf_ls = [tfs]
    Xs = [tf]
    Ys = [ba]
    priors = BC_priors
    
    Bs = fl.solve_ortho_direct(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS)
    Bs_str_l = []

    Bs_str_l.append('\t'.join(tfs))
    for gi in range(len(genes)):
        gene = genes[gi]
        regulators = Bs[0][:, gi]
        
        Bs_str_l.append(gene +'\t'+ '\t'.join(map(str, regulators)))
    f = file(outf, 'w')
    f.write('\n'.join(Bs_str_l))
    f.close()

def run_scr2(lamP, lamR, lamS, outf, sub1, sub2):
    #(ba,tf, genes, tfs) = load_bacteria('B_subtilis.csv','tfNames.txt',range(250))
    (ba,tf, genes, tfs) = load_B_anthracis(sub1, sub2)
    (BC_priors, sign) = ([], [])
    #(BC_priors, sign) = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    orth = load_orth('',['B_subtilis'])

    print ba.shape
    organisms = ['B_subtilis']
    gene_ls = [genes]
    tf_ls = [tfs]
    Xs = [tf]
    Ys = [ba]
    priors = BC_priors
    
    Bs = fl.solve_ortho_direct(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS)
    Bs_str_l = []

    Bs_str_l.append('\t'.join(tfs))
    for gi in range(len(genes)):
        gene = genes[gi]
        regulators = Bs[0][:, gi]
        
        Bs_str_l.append(gene +'\t'+ '\t'.join(map(str, regulators)))
    f = file(outf, 'w')
    f.write('\n'.join(Bs_str_l))
    f.close()


def load_priors(priors_fn, organism):    
    p = file(priors_fn)
    ps = p.read()
    psn = filter(len, ps.split('\n'))
    psnt = map(lambda x: x.split('\t'), psn)
    priors = map(lambda x: (fl.one_gene(x[0], organism), fl.one_gene(x[1], organism)), psnt)
    signs = map(lambda x: [-1,1][x[2]=='activation'], psnt)
    p.close()
    return (priors,signs)

#TODO pending seeing the orth file format
def load_orth(orth_fns, organisms):
    return []

def load_network(net_fn):
    f = file(net_fn)
    fs = f.read()
    fsl = filter(len, fs.split('\n'))
    fslt = map(lambda x: x.split('\t'), fsl)
    tfs = fslt[0]
    genes = map(lambda x: x[0], fslt[1:])
    #the network is written as the transpose of the matrix we want
    net = np.zeros((len(genes), len(tfs)))
    for g in range(len(genes)):
        targets = np.array(map(float, fslt[g+1][1:]))
        net[g, :] = targets
    return (net.T, genes, tfs)


def eval_prediction(net_fn, e, t, genes, tfs, metric):
    (net, genes, tfs) = load_network(net_fn)
    #(e, t, genes, tfs) = load_bacteria(expr_fn, tfs_fn, sub_conds)
    print net.shape
    err = fl.prediction_error(t, net, e, metric)
    return err
#looks at a network and returns an array of interaction weights for all priors in the network, along with an array containing the sign of the prior (+1, -1)
def check_prior_recovery(net_fn, priors_fn):
    (net, genes, tfs) = load_network(net_fn)
    (p, sign) = load_priors(priors_fn, '1')
    pcon = fl.priors_to_constraints(['1'], [genes],[tfs],p,0.5)
    prior_inter = []
    for con in pcon:
        prior_inter.append(net[con.c1.r, con.c1.c])
    
    return (np.array(prior_inter), np.array(sign))
            
#returns a list of tf gene correlations for each prior interaction, as well as the sign of each prior
def check_prior_corr(expr_fn, tfs_fn, priors_fn):
    (e, t, genes, tfs) = load_bacteria(expr_fn, tfs_fn)
    (p, s) = load_priors(priors_fn,'1')
    pcon = fl.priors_to_constraints(['1'], [genes],[tfs],p,0.5)

    prior_corr = []
    for con in pcon:
        tfi = con.c1.r
        gi = con.c1.c
        tf_expr = t[:, tfi]
        g_expr = e[:, gi]
        corr  = np.corrcoef(tf_expr, g_expr)[0,1]
        prior_corr.append(corr)
    return (prior_corr, s)

#creates a new expression matrix by joining two exp mats, with arbitrary names
#order of names in output is arbitrary
def join_expr_data(names1, names2, exp_a1, exp_a2):
    names = list(set(names1).intersection(set(names2)))
    name_to_ind = {names[x] : x for x in range(len(names))}
    #print name_to_ind
    def n_to_i(n):
        if n in name_to_ind:
            return name_to_ind[n]
        return -1
    
    exp_a = np.zeros((exp_a1.shape[0] + exp_a2.shape[0], len(names)))
    
    i=0
    #name_ind_to_name1_ind[i] maps an index in names to an index in names1
    name_ind_to_name1_ind = {n_to_i(names1[x]) : x for x in range(len(names1))}

    #name1_fr_inds[i] is the index in names1 that corresponds ti names[i]
    name1_fr_inds = map(lambda x: name_ind_to_name1_ind[x], range(len(names)))
    #array of the previous
    name1_fr_inds_a = np.array(name1_fr_inds)
    #copy into the array
    for r1 in range(exp_a1.shape[0]):
        exp_a[r1, :] = exp_a1[r1, name1_fr_inds_a]
    name_ind_to_name2_ind = {n_to_i(names2[x]) : x for x in range(len(names2))}
    name2_fr_inds = map(lambda x: name_ind_to_name2_ind[x], range(len(names)))
    name2_fr_inds_a = np.array(name2_fr_inds)
    for r2 in range(exp_a2.shape[0]):
        exp_a[r2+exp_a1.shape[0], :] = exp_a2[r2, name2_fr_inds_a]

    return (exp_a, names)


#quantile normalizes conditions AND scales to mean zero/unit variance
def normalize(exp_a, mean_zero = False):
    
    if mean_zero:
        exp_a = exp_a - exp_a.mean(axis=0)
    
    canonical_dist = np.sort(exp_a, axis=1).mean(axis=0)
    #if mean_zero:
    #    canonical_dist = canonical_dist - canonical_dist.mean()
    
    canonical_dist = canonical_dist / canonical_dist.std()
    
    exp_n_a = np.zeros(exp_a.shape)
    for r in range(exp_a.shape[0]):
        order = np.argsort(exp_a[r, :])
        exp_n_a[r, order] = canonical_dist

    return exp_n_a

#sub_conds is a bit weird here
def load_B_anthracis(sub_conds1 = [], sub_conds2 = []):
    (e1, t1, genes1, tfs1) = load_ba_iron(sub_conds1)
    (e2, t2, genes2, tfs2) = load_ba_timeseries(sub_conds2)
    (e, genes) = join_expr_data(genes1, genes2, e1, e2)
    (t, tfs) = join_expr_data(tfs1, tfs2, t1, t2)
    e = normalize(e, True)
    t = normalize(t, False)

    #(t, tfs) = join_expr_data(tfs1, tfs2, t1, t2)
    return (e, t, genes, tfs)
#this is a NAIVE loader.
#does not consider time series relationships
#returns (gene expr, tf expr, genes, tfs)
def load_ba_timeseries(sub_conds=[]):
    f = file('Normalized_data_RMA._txt')
    fs = f.read()
    fsn = filter(len, fs.split('\n'))
    fsnt = map(lambda x: x.split('\t'), fsn)
    conds = fsnt[0][1:]
    #first line is SCAN REF
    #second line is composite element REF
    #lines 3-end are data
    exp_mat_t = np.zeros((len(fsnt)-2, len(conds)))#first col gene
    genes = []
    f_tf = file('tfNamesAnthracis')
    f_tfs = f_tf.read()
    tfs = filter(len, f_tfs.split('\n'))


    for r in range(exp_mat_t.shape[0]):
        gene_str_full = fsnt[r+2][0]
        #gene name is 4th element, separated by ':'
        gene_str = gene_str_full.split(':')[3]
        gene_str = gene_str.replace('_pXO1_','').replace('_pXO2','')#what is this? dunno!
        expr = np.array(fsnt[r+2][1:])
        exp_mat_t[r, :] = expr
        genes.append(gene_str)

    #require that tfs be genes that we have data for!
    tfs = filter(lambda x: x in genes, tfs)

    tf_mat_t = np.zeros((len(tfs), len(conds)))

    gene_to_ind = {genes[x] : x for x in range(len(genes))}
    for ti in range(len(tfs)):
        gi = gene_to_ind[tfs[ti]]
        tf_mat_t[ti, :] = exp_mat_t[gi, :]
    exp_mat = exp_mat_t.T
    #exp_mat = (exp_mat_t - np.mean(exp_mat_t, axis=0)).T
    tf_mat = tf_mat_t.T
    if sub_conds == []:
        sub_conds = range(exp_mat.shape[0])
        
    sub_conds = np.array(sub_conds)
    return (exp_mat[sub_conds, :], tf_mat[sub_conds, :], genes, tfs)

def load_ba_iron(sub_conds = []):
    f = file('normalizedgeneexpressionvalues.txt')
    fs = f.read()
    fsn = filter(len, fs.split('\n'))
    fsnt = map(lambda x: x.split('\t'), fsn)
    conds = fsnt[0][1:]
    #first line is SCAN REF
    #second line is composite element REF
    #lines 3-end are data
    f_tf = file('tfNamesAnthracis')
    f_tfs = f_tf.read()
    tfs = filter(len, f_tfs.split('\n'))
    

    exp_mat_t = np.zeros((len(fsnt)-2, len(conds)))#first col gene
    genes = []
    #tfs = []
    for r in range(exp_mat_t.shape[0]):
        gene_str = fsnt[r+2][0]
        
        #gene name is 4th element, separated by ':'
        
        expr = np.array(fsnt[r+2][1:])
        exp_mat_t[r, :] = expr
        genes.append(gene_str)

    #require that tfs be genes that we have data for!
    tfs = filter(lambda x: x in genes, tfs)

    tf_mat_t = np.zeros((len(tfs), len(conds)))

    gene_to_ind = {genes[x] : x for x in range(len(genes))}
    for ti in range(len(tfs)):
        gi = gene_to_ind[tfs[ti]]
        tf_mat_t[ti, :] = exp_mat_t[gi, :]
    exp_mat = exp_mat_t.T
    #exp_mat = (exp_mat_t - np.mean(exp_mat_t, axis=0)).T
    tf_mat = tf_mat_t.T
    if sub_conds == []:
        sub_conds = range(exp_mat.shape[0])
        
    sub_conds = np.array(sub_conds)
    return (exp_mat[sub_conds, :], tf_mat[sub_conds, :], genes, tfs)



#this is a NAIVE loader.
#does not consider time series relationships
#returns (gene expr, tf expr, genes, tfs)
def load_B_subtilis(sub_conds=[]):
    (e, t, genes, tfs) =  load_bacteria('B_subtilis.csv', 'tfNames_subtilis.txt',sub_conds)
    e = normalize(e, True)
    t = normalize(t, False)
    return (e, t, genes, tfs)
#genes are mean 0
#not as general as I had hoped!
def load_bacteria(expr_fn, tfs_fn, sub_conds=[]):
    f = file(expr_fn)
    fs = f.read()
    fsl = filter(len, fs.split('\n'))
    fslc = map(lambda x: x.split(','), fsl)
    f.close()
    
    t = file(tfs_fn)
    ts = t.read()
    tfs = filter(len, ts.split('\n'))
    t.close()

    tfs_set = set(tfs)


    conds = fslc[0]
    genes = map(lambda x: x[0], fslc[1:])
    exp_mat_t = np.zeros((len(genes), len(conds)))
    for r in range(len(genes)):
        conds_f = map(float, fslc[1+r][1:])
        conds_a = np.array(conds_f)
        exp_mat_t[r, :] = conds_a

    tf_mat_t = np.zeros((len(tfs), len(conds)))
    gene_to_ind = {genes[x] : x for x in range(len(genes))}
    
    for ti in range(len(tfs)):
        gi = gene_to_ind[tfs[ti]]
        tf_mat_t[ti, :] = exp_mat_t[gi, :]

    exp_mat = exp_mat_t.T
    #exp_mat = exp_mat - exp_mat.mean(axis=0)
    #where is the right place to subtract the mean?
    tf_mat = tf_mat_t.T
    if sub_conds == []:
        sub_conds = range(exp_mat.shape[0])
        
    sub_conds = np.array(sub_conds)
    exp_mat = exp_mat[sub_conds, :]
    
    return (exp_mat, tf_mat[sub_conds, :], genes, tfs)

f = lambda x: run_scr(0.001,5,0,'wat4.tsv')
