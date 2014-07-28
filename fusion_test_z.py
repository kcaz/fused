import numpy as np
import itertools
import random
import fused_L2 as fr
fr.metric = 'mse'

#b_sparse is max num of nonzero entries in each col 
#returns b
def make_bs(b_dim,b_sparse):
    b = np.zeros(b_dim)
    for c in range(0, b_dim[1]):
        nnz = random.randrange(1, b_sparse+1)
        inds = random.sample(range(0, b_dim[0]), nnz)
        for r in inds:
            b[r, c] = np.random.randn()
    return b
    
#generates a matrix Y from the linear model specified by B.
#x is sampled uniformly from -1 to 1
#input xdims: dimensions of x
#input B: matrix specifying linear model
#input noise_std: std of noise!
#returns: (X, Y)
def generate_from_linear(xdims, B, noise_std):
    X = 1-2*np.random.random(xdims)
    Y = np.dot(X,B) + noise_std*np.random.randn(xdims[0], B.shape[1]) #note the inconsistency!!!!! AARGH!
    return (X, Y)    

def build_orth(genes1, genes2, max_grp_size, pct_fused, min_fuse_std, max_fuse_std, omap):
    random.shuffle(genes1)
    random.shuffle(genes2)
    
    amt_fused = np.round((len(genes1)+len(genes2))*pct_fused)
    ind1 = 0
    ind2 = 0
    while ind1 + ind2 < amt_fused:
        grp_size = random.randrange(2, max_grp_size+1)

        grp1_size = random.randrange(1, grp_size)
        grp2_size = grp_size - grp1_size
        #modify the group sizes to deal with not enough of one sub
        grp1_size = min(grp1_size, len(genes1)-ind1)
        grp2_size = min(grp2_size, len(genes2)-ind2) 
        
        for i in range(ind1, ind1 + grp1_size):
            for j in range(ind2, ind2 + grp2_size):
                sub1c = (0, genes1[i])
                sub2c = (1, genes2[j])
                
                if sub1c in omap:
                    omap[sub1c].append(sub2c)
                else:
                    omap[sub1c] = [sub2c]
                if sub2c in omap:
                    omap[sub2c].append(sub1c)
                else:
                    omap[sub2c] = [sub1c]
        ind1 += grp1_size
        ind2 += grp2_size

#rows and columns of b1/b2 are fused
def fuse_bs_orth(b1dims, b2dims, max_grp_size, pct_fused, min_fuse_std, max_fuse_std, sparse):
    b1 = np.nan * np.ones(b1dims)
    b2 = np.nan * np.ones(b2dims)
    tfs1 = range(b1.shape[0])
    tfs2 = range(b2.shape[0])    
    genes1 = range(b1.shape[0], b1.shape[1])
    genes2 = range(b2.shape[0], b2.shape[1])
    omap = dict()
    build_orth(tfs1, tfs2, max_grp_size, pct_fused, min_fuse_std, max_fuse_std, omap)
    build_orth(genes1, genes2, max_grp_size, pct_fused, min_fuse_std, max_fuse_std, omap)
    bs = [b1, b2]
    fusion_groups = []    
    #print 'map built'
    #fused to takes (sub, gene) row, (sub, gene) col
    def fused_to(orth_fr_r, orth_fr_c, acc_set):
        if not orth_fr_r in omap or not orth_fr_c in omap:
            return
        orth_rs = omap[orth_fr_r]
        orth_cs = omap[orth_fr_c]
        for orth_to_r in orth_rs:
            for orth_to_c in orth_cs:
                con = ((orth_fr_r, orth_fr_c), (orth_to_r, orth_to_c))
                #we need to make sure that the to section of this constraint refers to a coefficient that really exists! there may be fewer tfs than genes.
                to_sub = orth_to_r[0]
                if orth_to_r[1] >= bs[to_sub].shape[0]:
                    continue
                if not con in acc_set:
                    acc_set.add(con)
                    fused_to(orth_to_r, orth_to_c, acc_set)        
    orth_fr_r_l = []
    orth_fr_c_l = []
    #SET UP THESE DUDES TO ITERATE OVER
    for r in range(b1.shape[0]):
        orth_fr_r_l.append((0, r))
    for c in range(b1.shape[1]):    
        orth_fr_c_l.append((0, c))

    for r in range(b2.shape[0]):
        orth_fr_r_l.append((1, r))
    for c in range(b2.shape[1]):
        orth_fr_c_l.append((1, c))
    #print 'starting filling in'
    import time
    #print (len(orth_fr_r_l),len(orth_fr_c_l))
    #print len(orth_fr_r_l)*len(orth_fr_c_l)
    for orth_fr_r in orth_fr_r_l:
        for orth_fr_c in orth_fr_c_l:
            ti = time.time()
            #make sure that these are in the same subproblem!!!!
            (sub1, r) = orth_fr_r
            (sub2, c) = orth_fr_c
            if not sub1 == sub2:
                continue
                        
#check to make sure it's not set yet
            #NOTE this isn't really right, because the value of b could be set to zero, but it's not very likely
            if not np.isnan(bs[sub1][r, c]):
                continue            
            s = set()                        
            fused_to(orth_fr_r, orth_fr_c, s)
            coin = random.random()
            if coin < sparse:
                val = 0
                std = 0
            else:            
                val = np.random.randn()
                std = random.uniform(min_fuse_std, max_fuse_std)
            b1_coeffs = []
            b2_coeffs = []
            b_coeffs = [b1_coeffs, b2_coeffs]
            bs[sub1][r, c] = val + np.random.randn() * std
            for con in s:
                (coeff1, coeff2) = con
                ((sub11, r1), (sub12, c1)) = coeff1
                ((sub21, r2), (sub22, c2)) = coeff2
                #only consider constraints going one way
                if sub11 == sub21 or sub11 > sub21:
                    continue #no fusion within groups FOR NOW!
                #is sock
                bs[sub11][r1, c1] = val + np.random.randn() * std
                bs[sub21][r2, c2] = val + np.random.randn() * std
                #the format for fusion groups is a list of tuples, containing lists of (r, c) tuples for sub1 in the first position and lists of (r, c) tuples for sub 2 in the second. these coefficients are fused           
                b_coeffs[sub11].append((r1, c1))
                b_coeffs[sub21].append((r2, c2))
            if len(b1_coeffs) or len(b2_coeffs):
                #don't append empty groups
                fusion_groups.append((b1_coeffs, b2_coeffs))
            
    return (b1, b2, fusion_groups)
                
                
#max_grp_size is the maximum size of fusion group
#pct_fused is the proportion of the entries in b1 + b2 that become fused
#min_fuse_std, max_fuse_std range of std for fusion error
def fuse_bs(b1, b2, max_grp_size, pct_fused, min_fuse_std, max_fuse_std):
    fusiongroups = []
    amt_fused = np.round((b1.shape[0]*b1.shape[1] + b2.shape[0]*b2.shape[1])*pct_fused)
    
    b1_inds = list(itertools.product(range(b1.shape[0]), range(b1.shape[1])))
    b2_inds = list(itertools.product(range(b2.shape[0]), range(b2.shape[1])))
    b1_inds = filter(lambda x: b1[x[0],x[1]] != 0, b1_inds)
    b2_inds = filter(lambda x: b2[x[0],x[1]] != 0, b2_inds)
    amt_fused = np.round(pct_fused * (len(b1_inds)+len(b2_inds)))
    #random.shuffle(b1_inds)
    #random.shuffle(b2_inds)

    ind1 = 0
    ind2 = 0

    while ind1 + ind2 < amt_fused: 
        grp_size = random.randrange(2, max_grp_size+1)
        grp1_size = random.randrange(1, grp_size)
        grp2_size = grp_size - grp1_size
        
        b1_sel = b1_inds[ind1:(ind1+grp1_size)]
        b2_sel = b2_inds[ind2:(ind2+grp2_size)]
        ind1 += grp1_size
        ind2 += grp2_size

        fusiongroups.append((b1_sel, b2_sel))
        
        
        val = np.random.randn()
        std = random.uniform(min_fuse_std, max_fuse_std)
        
        for b1_ind in b1_sel:
            (r, c) = b1_ind
            b1[r, c] = val + np.random.randn()*std
        for b2_ind in b2_sel:
            (r, c) = b2_ind
            b2[r, c] = val+np.random.randn()*std

    for i in range(ind1, len(b1_inds)):
        (r, c) = b1_inds[i]
        val = np.random.randn()
        std = random.uniform(min_fuse_std, max_fuse_std)
        b1[r, c] = val + np.random.randn()*std

    for i in range(ind2, len(b2_inds)):
        (r, c) = b2_inds[i]
        val = np.random.randn()
        std = random.uniform(min_fuse_std, max_fuse_std)
        b2[r, c] = val + np.random.randn()*std

    return (b1, b2, fusiongroups)


#left as an exercise to the reader
# returns a list of indices (2-element tuples), each of which refers to a prior on a nonzero entry of b 
def messwpriors(b, falsepos, falseneg):
    priors = []
    for r in range(b.shape[0]):
        for c in range(b.shape[1]):
            if b[r, c] == 0 and np.random.rand() < falsepos:
                priors.append((r, c))
            if b[r, c] != 0 and np.random.rand() > falseneg:
                priors.append((r, c))
    return priors


def pred_err_grps(B, X_lo, Y_lo):
    errs = np.zeros(Y_lo.shape[1])
    predY = np.array(np.dot(X_lo, B))
    for c in range(Y_lo.shape[1]):
        err = np.mean((predY[:, c] - Y_lo[:,c])**2)
        errs[c] = err    
    return errs.mean()


#xsamples = x.shape[0]
def test_linearBs(b1, b2, fusiongroups, xsamples1, xsamples2, noise_std1, noise_std2, p_falsep, p_falseneg, lamP, lamR, lamS):
    TFs = [map(str, range(b1.shape[0])), map(str, range(b2.shape[0]))]
    Gs = [map(str, range(b1.shape[1])), map(str, range(b2.shape[1]))]
    xdims1 = (xsamples1, b1.shape[0])
    xdims2 = (xsamples2, b2.shape[0])
    (x1, y1) = generate_from_linear(xdims1, b1, noise_std1)
    (x2, y2) = generate_from_linear(xdims2, b2, noise_std2)
    (x1test, y1test) = generate_from_linear(xdims1, b1, noise_std1)
    (x2test, y2test) = generate_from_linear(xdims2, b2, noise_std2)
    p1 = messwpriors(b1, p_falsep, p_falseneg)
    p2 = messwpriors(b2, p_falsep, p_falseneg)
    priorset = [p1, p2]
    #changed
    bs_solve = fr.solve_group_direct([x1, x2], [y1, y2], fusiongroups, priorset, lamP, lamR, lamS)
    err1 = pred_err_grps(bs_solve[0], x1test, y1test)
    err2 = pred_err_grps(bs_solve[1], x2test, y2test)

    
    return (bs_solve, err1, err2)


def benchmark(lamP, lamR, lamS, b1dim, b2dim, maxgroupsize, pct_fused, minfusestd, maxfusestd, xsamples1, xsamples2, noise1, noise2, p_falsep, p_falsen, sparse, it):
    watdict = dict()
    for R in lamR:
        for S in lamS:
            for P in lamP:
                wat = []                
                #wat = 0
                for i in range(it):
                    #b1 = make_bs(b1dim,b1spars)
                    #b2 = make_bs(b2dim,b2spars)
                    #(b1f, b2f, o) = fuse_bs(b1, b2, maxgroupsize, pct_fused, minfusestd, maxfusestd)
                    (b1f, b2f,o) = fuse_bs_orth(b1dim,b2dim, maxgroupsize, pct_fused, minfusestd, maxfusestd, sparse)
                    
                    (bs_solve, err1, err2) = test_linearBs(b1f, b2f, o, xsamples1, xsamples2, noise1, noise2, p_falsep, p_falsen, P, R, S)
                    wat.append(np.mean([err1.mean(), err2.mean()]))                    
                    #wat += np.mean([err1.mean(), err2.mean()])
                watdict[(P, R, S)] = (np.mean(wat), np.std(wat)/it**0.5)#wat/it
    return watdict


#wd = benchmark(lamP = [1], lamR = [1], lamS = [0.5], b1dim = (2,2), b1spars = 2, b2dim = (2,2), b2spars = 2, maxgroupsize = 2, pct_fused = 1.0, minfusestd = 0.0, maxfusestd = 0.0, xsamples1 = 1, xsamples2 = 1, noise1 = 0.1, noise2 = 0.1, p_falsep = 0.0, p_falsen = 0.0, it=20)
