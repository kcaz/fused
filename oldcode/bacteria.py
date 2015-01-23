import numpy as np
import fused_L2 as fl
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
iron_conds = 24
timeseries_conds = 51
subtilis_conds = 359

def write_bs(genes, tfs, B, outf):
    Bs_str_l = []
    Bs_str_l.append('\t'.join(tfs))
    for gi in range(len(genes)):
        gene = genes[gi]
        regulators = B[:, gi]
        
        Bs_str_l.append(gene +'\t'+ '\t'.join(map(str, regulators)))
    f = file(outf, 'w')
    f.write('\n'.join(Bs_str_l))
    f.close()

def run_both(lamP, lamR, lamS, outf, sub_s, sub_i, sub_t):
    (bs_e, bs_t , bs_genes, bs_tfs) = load_B_subtilis(sub_s)
    (BS_priors, sign) = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    (ba_e, ba_t, ba_genes, ba_tfs) = load_B_anthracis(sub_i, sub_t)
    (BA_priors, sign) = ([], [])

    Xs = [bs_t, ba_t]
    Ys = [bs_e, ba_e]
    priors = BS_priors + BA_priors
    
    orth = load_orth('bs_ba_ortho_804',['B_anthracis','B_subtilis'])
    #orth = load_orth('',['B_subtilis'])
    organisms = ['B_subtilis','B_anthracis']
    #ortht = random_orth(bs_tfs, ba_tfs, organisms, 250)
    #orthg = random_orth(bs_genes, ba_genes, organisms, 2500)
    #orth = ortht+orthg
    #print orth
    #return
    gene_ls = [bs_genes, ba_genes]
    tf_ls = [bs_tfs, ba_tfs]

    Bs = fl.solve_ortho_direct(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, 'no_adjust', lamP, lamR, lamS)
    write_bs(bs_genes, bs_tfs, Bs[0], outf+'_subtilis')
    write_bs(ba_genes, ba_tfs, Bs[1], outf+'_anthracis')

def run_both_adjust(lamP, lamR, lamS, outf, sub_s, sub_i, sub_t):
    (bs_e, bs_t , bs_genes, bs_tfs) = load_B_subtilis(sub_s)
    (BS_priors, sign) = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    (ba_e, ba_t, ba_genes, ba_tfs) = load_B_anthracis(sub_i, sub_t)
    (BA_priors, sign) = ([], [])
    Xs = [bs_t, ba_t]
    Ys = [bs_e, ba_e]
    priors = BS_priors + BA_priors
    orth = load_orth('bs_ba_ortho_804',['B_anthracis','B_subtilis'])
    #orth = load_orth('',['B_subtilis'])
    organisms = ['B_subtilis','B_anthracis']
    #ortht = random_orth(bs_tfs, ba_tfs, organisms, 250)
    #orthg = random_orth(bs_genes, ba_genes, organisms, 2500)
    #orth = ortht+orthg
    #print orth
    #return
    gene_ls = [bs_genes, ba_genes]
    tf_ls = [bs_tfs, ba_tfs]
    Bs = fl.solve_ortho_direct(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, 'yes', lamP, lamR, lamS)
    write_bs(bs_genes, bs_tfs, Bs[0], outf+'_subtilis')
    write_bs(ba_genes, ba_tfs, Bs[1], outf+'_anthracis')


def run_both_refit(lamP, lamR, lamS, outf, sub_s, sub_i, sub_t, it, k):
    (bs_e, bs_t , bs_genes, bs_tfs) = load_B_subtilis(sub_s)
    (BS_priors, sign) = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    (ba_e, ba_t, ba_genes, ba_tfs) = load_B_anthracis(sub_i, sub_t)
    (BA_priors, sign) = ([], [])

    Xs = [bs_t, ba_t]
    Ys = [bs_e, ba_e]
    priors = BS_priors + BA_priors
    
    orth = load_orth('bs_ba_ortho_804',['B_anthracis','B_subtilis'])
    #orth = load_orth('',['B_subtilis'])
    organisms = ['B_subtilis','B_anthracis']
    #ortht = random_orth(bs_tfs, ba_tfs, organisms, 250)
    #orthg = random_orth(bs_genes, ba_genes, organisms, 2500)
    #orth = ortht+orthg
    #print orth
    #return
    gene_ls = [bs_genes, ba_genes]
    tf_ls = [bs_tfs, ba_tfs]

    Bs = fl.solve_ortho_direct_refit(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS, it, k)
    write_bs(bs_genes, bs_tfs, Bs[0], outf+'_subtilis')
    write_bs(ba_genes, ba_tfs, Bs[1], outf+'_anthracis')
    
#with saturation
def run_both_scad(lamP, lamR, lamS, outf, sub_s, sub_i, sub_t, it,k, it_s):
    (bs_e, bs_t , bs_genes, bs_tfs) = load_B_subtilis(sub_s)
    (BS_priors, sign) = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    (ba_e, ba_t, ba_genes, ba_tfs) = load_B_anthracis(sub_i, sub_t)
    (BA_priors, sign) = ([], [])

    Xs = [bs_t, ba_t]
    Ys = [bs_e, ba_e]
    priors = BS_priors + BA_priors
    
    orth = load_orth('bs_ba_ortho_804',['B_anthracis','B_subtilis'])
    #orth = load_orth('',['B_subtilis'])
    organisms = ['B_subtilis','B_anthracis']
    #ortht = random_orth(bs_tfs, ba_tfs, organisms, 250)
    #orthg = random_orth(bs_genes, ba_genes, organisms, 2500)
    #orth = ortht+orthg
    #print orth
    #return
    gene_ls = [bs_genes, ba_genes]
    tf_ls = [bs_tfs, ba_tfs]


    Bs = fl.solve_ortho_scad_refit(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS, it, k, it_s)
    write_bs(bs_genes, bs_tfs, Bs[0], outf+'_subtilis')
    write_bs(ba_genes, ba_tfs, Bs[1], outf+'_anthracis')



def run_scr(lamP, lamR, lamS, outf, sub):
    #(ba,tf, genes, tfs) = load_bacteria('B_subtilis.csv','tfNames.txt',range(250))
    (ba,tf, genes, tfs) = load_B_subtilis(sub)
    (BC_priors, sign) = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    orth = []#load_orth('',['B_subtilis'])

    print ba.shape
    organisms = ['B_subtilis']
    gene_ls = [genes]
    tf_ls = [tfs]
    Xs = [tf]
    Ys = [ba]
    priors = BC_priors
    
    Bs = fl.solve_ortho_direct_refit(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS, 0, 100)
    #Bs = fl.solve_ortho_direct(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS)
    Bs_str_l = []

    Bs_str_l.append('\t'.join(tfs))
    for gi in range(len(genes)):
        gene = genes[gi]
        regulators = Bs[0][:, gi]
        
        Bs_str_l.append(gene +'\t'+ '\t'.join(map(str, regulators)))
    f = file(outf, 'w')
    f.write('\n'.join(Bs_str_l))
    f.close()

def run_scr2(lamP, lamR, lamS, outf, subi, subt):
    #(ba,tf, genes, tfs) = load_bacteria('B_subtilis.csv','tfNames.txt',range(250))
    (ba,tf, genes, tfs) = load_B_anthracis(subi, subt)
    (BC_priors, sign) = ([], [])
    #(BC_priors, sign) = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    orth = []

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



#this returns pairs of genes, not coefficients, along with interaction valences

#that's weird, but was done so decouple loading priors from needing to know the tfs
def load_priors(priors_fn, organism):    
    p = file(priors_fn)
    ps = p.read()
    psn = filter(len, ps.split('\n'))
    psnt = map(lambda x: x.split('\t'), psn)
    priors = map(lambda x: (fl.one_gene(x[0], organism), fl.one_gene(x[1], organism)), psnt)
    signs = map(lambda x: [-1,1][x[2]=='activation'], psnt)
    p.close()
    return (priors,signs)


#returns a list of random 2 element ortho groups for testing purposes
def random_orth(genes1, genes2, organisms, n_orth):
    orths = []
    for i in range(n_orth):
        g1 = genes1[int(np.floor(random.random() * len(genes1)))]
        g2 = genes2[int(np.floor(random.random() * len(genes2)))]
        orth_group = [fl.one_gene(g1, organisms[0]), fl.one_gene(g2, organisms[1])]
        orths.append(orth_group)
    return orths
        
#TODO pending seeing the orth file format
def load_orth(orth_fn, organisms):
    f = file(orth_fn)
    fs = f.read()
    fsn = filter(len, fs.split('\n'))
    fsnt = map(lambda x: x.split('\t'), fsn)
    #print fsn
    orths = []
    for o in fsnt:
        #print o
        orths.append([fl.one_gene(name=o[0],organism=organisms[0]), fl.one_gene(name=o[1], organism=organisms[1])])
    return orths

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
#AND DOES NOT divides expression matrices by the square root of the sample size
#CHANGED! mean subtract after quantile normalizing
def normalize(exp_a, mean_zero = False):
    
    
    canonical_dist = np.sort(exp_a, axis=1).mean(axis=0)
    #if mean_zero:
    #    canonical_dist = canonical_dist - canonical_dist.mean()
    
    canonical_dist = canonical_dist / canonical_dist.std()
    
    exp_n_a = np.zeros(exp_a.shape)
    for r in range(exp_a.shape[0]):
        order = np.argsort(exp_a[r, :])
        exp_n_a[r, order] = canonical_dist
    #exp_n_a / np.sqrt(exp_n_a.shape[0])
    if mean_zero:
        exp_n_a = exp_n_a - exp_n_a.mean(axis=0)

    return exp_n_a

#sub_conds is a bit weird here
def load_B_anthracis(subi = [], subt = []):
    (e1, t1, genes1, tfs1) = load_ba_iron()
    (e2, t2, genes2, tfs2) = load_ba_timeseries()
    
    (e, genes) = join_expr_data(genes1, genes2, e1, e2)
    (t, tfs) = join_expr_data(tfs1, tfs2, t1, t2)
    #drop the _at from the name
    genes = map(lambda x: x.replace('_at',''), genes)
    tfs = map(lambda x: x.replace('_at',''), tfs)
    e = normalize(e, True)
    t = normalize(t, False)
    #if empty, use everything
    if not len(subi):
        subi = range(e1.shape[0])
    if not len(subt):
        subt = range(e2.shape[0])
    subi = np.array(subi)
    subt = e1.shape[0] + np.array(subt)
    
    
    
    e = e[np.concatenate((subi, subt)), :]
    
    t = t[np.concatenate((subi, subt)), :]
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
#changed to normalize after subsetting conditions
def load_B_subtilis(sub_conds=[]):
    (e, t, genes, tfs) =  load_bacteria('B_subtilis.csv', 'tfNames_subtilis.txt',[])
    e = normalize(e, True)
    t = normalize(t, False)
    sub_conds = np.array(sub_conds)
    if len(sub_conds):
        e = e[sub_conds,:]
        t = t[sub_conds,:]
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


def order_corr_mat(cmat, init):

    cols = set(range(cmat.shape[1]))
    curr = init
    cols.remove(init)
    order = [init, ]

    while len(cols):
        best_c = 0
        best_val = np.inf
        for c in cols:
            nrm = np.linalg.norm(cmat[:, c] - cmat[:,curr]) 
            if nrm < best_val:
                best_c = c
                best_val = nrm
        order.append(best_c)
        cols.remove(best_c)
        curr = best_c
        
    return order

#no args!
def betas_fused_visualize(net_s, net_a, orth):
    from matplotlib import pyplot as plt
    (ns, gs, ts) = load_network(net_s)
    (na, ga, ta) = load_network(net_a)
    #we want to enumerate the constraints, fused_L2 can do that
    constraints = fl.orth_to_constraints(['B_subtilis','B_anthracis'], [gs, ga], [ts, ta], orth, 0)
    
    coeffs_s = []
    coeffs_a = []
    for con in constraints:
        
        if con.c1.sub == 1:
            continue
        beta_s = ns[con.c1.r, con.c1.c]
        beta_a = na[con.c2.r, con.c2.c]
        coeffs_s.append(beta_s)
        coeffs_a.append(beta_a)
    print np.corrcoef(coeffs_s, coeffs_a)
    plt.scatter(coeffs_s, coeffs_a)
    plt.xlabel('B subtilis')
    plt.ylabel('B anthracis')
    plt.show()
    cs = np.array(coeffs_s)
    ca = np.array(coeffs_a)
    #plt.hist(np.abs(cs-ca)/(0.5*(np.abs(cs)+np.abs(ca))), bins=50)
    #plt.show()

def betas_fused_corr(net_s, net_a, orth):

    (ns, gs, ts) = load_network(net_s)
    (na, ga, ta) = load_network(net_a)
    #we want to enumerate the constraints, fused_L2 can do that
    constraints = fl.orth_to_constraints(['B_subtilis','B_anthracis'], [gs, ga], [ts, ta], orth, 0)
    
    coeffs_s = []
    coeffs_a = []
    for con in constraints:
        
        if con.c1.sub == 1:
            continue
        beta_s = ns[con.c1.r, con.c1.c]
        beta_a = na[con.c2.r, con.c2.c]
        coeffs_s.append(beta_s)
        coeffs_a.append(beta_a)
    return np.corrcoef(coeffs_s, coeffs_a)[0,1]
    


def corr_visualize(genes1, genes2, exp1, exp2, organisms, orth):
    from matplotlib import pyplot as plt
    def ind_rc(m, inds):
        return m[:, inds][inds, :]

    orth12 = dict()
    orth21 = dict()
    for ogroup in orth:
        
        org1 = filter(lambda x: x.organism == organisms[0], ogroup)
        org2 = filter(lambda x: x.organism == organisms[1], ogroup)
        
#take only the first pair in each orthology group
        if len(org1) and len(org2):
            orth12[org1[0].name] = org2[0].name
            orth21[org2[0].name] = org1[0].name


    
    genes1o = filter(lambda x: x in orth12, genes1)
    genes2o = filter(lambda x: x in orth21, genes2)
    
    genes1o_s = set(genes1o)
    genes2o_s = set(genes2o)
    genes1o = filter(lambda x: orth12[x] in genes2o_s, genes1o)
    genes2o = filter(lambda x: orth21[x] in genes1o_s, genes2o)
    genes1o_s = set(genes1o)
    genes2o_s = set(genes2o)


    #print len(genes1o)
    #print len(genes2o)
    
    gi1 = {genes1o[x]:x for x in range(len(genes1o))}
    gi2 = {genes2o[x]:x for x in range(len(genes2o))}
    
    
    org2_order = np.array(map(lambda x: gi1[orth21[x]], genes2o))
    
    
    in_orth1 = np.array(map(lambda x: x in genes1o_s, genes1))
    in_orth2 = np.array(map(lambda x: x in genes2o_s, genes2))
    
    exp1 = exp1[:, in_orth1]
    exp2 = exp2[:, in_orth2]
    #exp2 = exp2[:, org2_order]
    #return (genes1o, genes2o, org2_order, orth21)
    #print org2_order

    #plt.matshow(exp1)
    #plt.show()
    cmat1 = np.corrcoef(exp1, rowvar=False)
    cmat2 = np.corrcoef(exp2, rowvar=False)

    
    cmat2 = ind_rc(cmat2, org2_order)
    
#cmat1 = ind_rc(cmat1, org2_order)
    

    nice_order = order_corr_mat(cmat1,int(cmat1.shape[1]*random.random()))
    #nice_order = org2_order
    cmat1 = ind_rc(cmat1, nice_order)
   # 
    cmat2 = ind_rc(cmat2, nice_order)
    #plt.subplot(121)
    plt.matshow(cmat1)
    #plt.subplot(122)
    
    plt.matshow(cmat2)
    plt.show()

    
    cflat1 = cmat1.ravel().copy()
    cflat2 = cmat2.ravel().copy()
    #print sum(in_orth1)
    #print sum(in_orth2)
    #print cmat1.shape
    #print cmat2.shape
    #print np.corrcoef(cflat1, cflat2)
    #print exp1.shape
    #print exp2.shape
    random.shuffle(cflat1)
    random.shuffle(cflat2)
    plt.scatter(cflat1[0:1000], cflat2[0:1000])
    plt.xlabel('B subtilis correlation')
    plt.ylabel('B anthracis correlation')
    plt.show()
    
    return (cmat1, cmat2)

#scores: R
#labels: 0/1
def prc(scores, labels):
    i_scores = np.argsort(-1*scores)
    s_labels = np.array(labels)[i_scores]
    cs = np.cumsum(s_labels)
    true_positive = np.sum(labels)
    precision = cs / np.arange(1, len(cs)+1)
    
    recall = cs / true_positive
    
    precision2 = []
    recall2 = []
    #prev = -np.inf
    #for sci in range(1, len(scores[i_scores])):
    #    if scores[i_scores[sci]] != prev:
    #        precision2.append(precision[sci-1])
    #        recall2.append(recall[sci-1])
    #    prev = scores[i_scores[sci]]
        
    return (precision, recall)

def eval_network_pr(net, genes, tfs, priors):
    from matplotlib import pyplot as plt
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
    
    #plt.plot(recall, precision)
    #plt.xlabel('recall')
    #plt.ylabel('precision')
    #plt.title('B subtilis alone, no refitting')
    #plt.show()
    
    return aupr


def eval_network_roc(net, genes, tfs, priors):
    from matplotlib import pyplot as plt
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
    plt.plot(fpr, tpr)
    plt.show()
    print auroc
    return (labels, scores)
        
f = lambda x: run_scr(0.001,5,0,'wat4.tsv')
