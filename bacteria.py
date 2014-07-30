import numpy as np
import fused_L2 as fl

def run_scr(lamP, lamR, lamS, outf):
    (ba,tf, genes, tfs) = load_bacteria('B_subtilis.csv','tfNames.txt')
    BC_priors = load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    orth = load_orth('',['B_subtilis'])


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
    p.close()
    return priors

#TODO pending seeing the orth file format
def load_orth(orth_fns, organisms):
    return []

def load_bacteria(expr_fn, tfs_fn):
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


    return (exp_mat_t.T, tf_mat_t.T, genes, tfs)

run_scr(0.1,50,0,'wat2.tsv')
