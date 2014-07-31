import numpy as np
import fused_L2 as fl

def run_scr(lamP, lamR, lamS, outf):
    (ba,tf, genes, tfs) = load_bacteria('B_subtilis.csv','tfNames.txt',range(250))
    (BC_priors, sign) = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    orth = load_orth('',['B_subtilis'])

    print ba.shape
    organisms = ['B_subtilis']
    gene_ls = [genes]
    tf_ls = [tfs]
    Xs = [tf]
    Ys = [ba]
    priors = BC_priors
    #Bs=[np.random.randn(100,10000)]
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


def eval_prediction(net_fn, expr_fn, tfs_fn, metric, sub_conds=[]):
    (net, genes, tfs) = load_network(net_fn)
    (e, t, genes, tfs) = load_bacteria(expr_fn, tfs_fn, sub_conds)
    
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


#genes are mean 0
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

    exp_mat = (exp_mat_t - np.mean(exp_mat_t, axis=0)).T
    tf_mat = tf_mat_t.T
    if sub_conds == []:
        sub_conds = range(exp_mat.shape[0])
        
    sub_conds = np.array(sub_conds)
    return (exp_mat[sub_conds, :], tf_mat[sub_conds, :], genes, tfs)

f = lambda x: run_scr(0.001,5,0,'wat4.tsv')
