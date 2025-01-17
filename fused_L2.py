import numpy as np
from numpy import linalg
import random
import collections
import time
import scipy.sparse
import scipy.sparse.linalg
coefficient = collections.namedtuple('coefficient',['sub','r','c'])

#c2 == None is a constant constraint
constraint = collections.namedtuple('constraint',['c1','c2','lam'])


one_gene = collections.namedtuple('gene',['name','organism'])

#returns a list of lists of all pairs of entries from l, where the first entry in the pair occurs before the second in l
def all_pairs(l):
    pl = []
    for li1 in range(len(l)):
        for li2 in range(li1+1, len(l)):
           pl.append([l[li1], l[li2]])
    return pl

class dictl(dict):
    def __getitem__(self, i):
        if not i in self:
            self[i] = []
        return super(dictl, self).__getitem__(i)

def get_ridge_constraints(organisms, gene_ls, tf_ls, priors, lam, lamR):
    priors = priors_to_constraints(organisms, gene_ls, tf_ls, priors, lam)
    priors_s = set(map(lambda x: x.c1, priors))
    ridge_cons = []
    for sub in range(len(organisms)):
        for tfi in range(len(tf_ls[sub])):
            for gi in range(len(gene_ls[sub])):
                coeff = coefficient(sub, tfi, gi)
                if not coeff in priors_s:
                   ridge_cons.append(constraint(coeff, None, lamR))
    return priors + ridge_cons      

def priors_to_constraints(organisms, gene_ls, tf_ls, priors, lam):
    org_to_ind = {organisms[x] : x for x in range(len(organisms))}
    gene_to_inds = map(lambda o: {gene_ls[o][x] : x for x in range(len(gene_ls[o]))}, range(len(organisms)))
    tf_to_inds = map(lambda o: {tf_ls[o][x] : x for x in range(len(tf_ls[o]))}, range(len(organisms)))
    constraints = []
    for prior in priors:
        (gene1, gene2) = prior
        sub1 = org_to_ind[gene1.organism]
        sub2 = org_to_ind[gene2.organism]
        if not sub1 == sub2:
            print '!?!?!?!?!?'
            continue
        tfi = tf_to_inds[sub1][gene1.name]
        gi = gene_to_inds[sub2][gene2.name]
        constr = constraint(coefficient(sub1, tfi, gi), None, lam)
        constraints.append(constr)
    return constraints

def quad(a, b, c):
    x1 = 0.5*(-b + (b**2 - 4*a*c)**0.5)/a
    x2 = 0.5*(-b - (b**2 - 4*a*c)**0.5)/a
    
    return (x1, x2)

#also introduces new ridge constraints as appropriate
def adjust_vol2(Xs, cols, fuse_cons, ridge_cons, lamR):
    if len(fuse_cons) == 0:    
        return (fuse_cons, ridge_cons)
        #we're building a canonical ordering over the columns for the current set of columns
    counter = 0
    b_to_n = dict()
    n_to_b = dict()    
    for co in cols:
        sub = co.sub
        for r in range(Xs[sub].shape[1]):
            b_to_n[(sub, co.c, r)] = counter
            n_to_b[counter] = (sub, co.c, r)
            counter += 1
    N = counter
    Einv = np.zeros((N, N))
    for n in range(N):
        Einv[n, n] = lamR
    for ridge_con in ridge_cons:
        n = b_to_n[(ridge_con.c1.sub, ridge_con.c1.c, ridge_con.c1.r)]
        Einv[n, n] = ridge_con.lam
    Einv_noS = Einv.copy()
    
    adj_rows = set()
    for fuse_con in fuse_cons:
        n = b_to_n[(fuse_con.c1.sub, fuse_con.c1.c, fuse_con.c1.r)]
        m = b_to_n[(fuse_con.c2.sub, fuse_con.c2.c, fuse_con.c2.r)]
        Einv[n, m] += -fuse_con.lam
        Einv[m, n] += -fuse_con.lam
        Einv[n, n] += fuse_con.lam
        Einv[m, m] += fuse_con.lam
        adj_rows.add(m)
        adj_rows.add(n)

    Einv_noS = Einv_noS/lamR
    d2 = np.linalg.det(Einv_noS)   
    Einv = Einv/lamR
    d1 = np.linalg.det(Einv)

    import matplotlib.pyplot as plt
    #plt.matshow(Einv)
    #plt.show()
    #plt.plot(np.diag(Einv))
    #plt.show()
    #print (d2, d1)    
    #think about left multiplying by an identity matrix with some constant c in entries that show up in adj_rows in order to equate determinants
    c = (d2/d1)**(1.0/len(adj_rows))
    
    if np.isnan(c):
        print Einv
        #print [Einv[i][i] for i in range(len(matrix[0]))]
        plt.plot(np.diag(Einv))
        plt.show()
        print(d1, d2, len(adj_rows), c)
    fuse_adj = []
    ridge_adj = []
    for fuse_con in fuse_cons:
        n = b_to_n[(fuse_con.c1.sub, fuse_con.c1.c, fuse_con.c1.r)]
        m = b_to_n[(fuse_con.c2.sub, fuse_con.c2.c, fuse_con.c2.r)]
        if n in adj_rows or m in adj_rows:
            con = constraint(fuse_con.c1,fuse_con.c2,fuse_con.lam*c)            
            fuse_adj.append(con)
        else:
            con = constraint(fuse_con.c1,fuse_con.c2,fuse_con.lam)
            fuse_adj.append(con)
    for ridge_con in ridge_cons:
        n = b_to_n[(ridge_con.c1.sub, ridge_con.c1.c, ridge_con.c1.r)]
        if n in adj_rows:
            con = constraint(ridge_con.c1, ridge_con.c2, ridge_con.lam*c)
            ridge_adj.append(con)
        else:
            con = constraint(ridge_con.c1, ridge_con.c2, ridge_con.lam)
            ridge_adj.append(con)

    #now introduce new constraints that were previously default
    ridge_coeff_s = set(map(lambda x: x.c1, ridge_cons))
    for n in adj_rows:
        (sub, c, r) = n_to_b[n]
        coeff = coefficient(sub, r, c)
        if not coeff in ridge_coeff_s:
            con = constraint(coeff, None, lamR*c)
            ridge_adj.append(con)        
        
    return (fuse_adj, ridge_adj)


def adjust_vol(Xs, cols, fuse_cons, ridge_cons, lamR):
    return (fuse_cons, ridge_cons)
        #we're building a canonical ordering over the columns for the current set of columns
    counter = 0
    sub_offs = dict()
    for co in cols:
        sub = co.sub
        if not (co.sub, co.c) in sub_offs:
            sub_offs[(sub, co.c)] = counter
            counter += Xs[sub].shape[1]
    N = counter
    Einv = np.zeros((N, N))
    for n in range(N):
        Einv[n, n] = lamR
    for ridge_con in ridge_cons:
        n = sub_offs[(ridge_con.c1.sub, ridge_con.c1.c)] + ridge_con.c1.r
        Einv[n, n] = ridge_con.lam
    d2 = np.linalg.det(Einv)
    for fuse_con in fuse_cons:
        n = sub_offs[(fuse_con.c1.sub, fuse_con.c1.c)] + fuse_con.c1.r
        m = sub_offs[(fuse_con.c2.sub, fuse_con.c2.c)] + fuse_con.c2.r
        Einv[n, m] += -2*fuse_con.lam
        Einv[m, n] += -2*fuse_con.lam
        Einv[n, n] += fuse_con.lam
        Einv[m, m] += fuse_con.lam
    d1 = np.linalg.det(Einv)
    c = (d2/d1)**(1.0/N)
    
    if np.isnan(c):
        print Einv
        print Einv_nos
        print(d1, d2, c)
    fuse_adj = map(lambda x: constraint(x.c1, x.c2, x.lam*c), fuse_cons)
    ridge_adj = map(lambda x: constraint(x.c1,x.c2, x.lam*c), ridge_cons)
    return (fuse_adj, ridge_adj, c*lamR)


def adjust(lamR1, lamR2, lamS):
        #this code is written strangely, to match the google doc
    if lamS == 0:
        return (lamR1, lamR2) #avoid numerical problems
    lamR1 = float(lamR1)
    lamR2 = float(lamR2)
    lamS = float(lamS)#just in case
    a = 1
    b = lamS - lamR1
    c = lamS
    d = -lamR1*lamS
    
    e = 1
    f = lamS
    g = lamS - lamR2
    h = -lamR2*lamS
    
    qa1 = (g*a - e*c)
    qb1 = (b*g + h*a - e*d - f*c)
    qc1 = (h*b - f*d)
    
    qa2 = (f*a - e*b)
    qb2 = (f*c + h*a - e*d - g*b)
    qc2 = (h*c - g*d)
    
    lamRa1 = max(quad(qa1, qb1, qc1))
    lamRa2 = max(quad(qa2, qb2, qc2))
    
    return (lamRa1, lamRa2)


#for each fusion constraint, introduces a ridge constraint to compensate for over-regularization. Equalizes variance of priors in case of 1-1 fusion
def adjust_ridge_fused(fuse_con, ridge_con, lamR):
    #TEMPORARILY DOES NOTHING
    return ridge_con
    #this dictionary maps a coefficient to the fusion constraint that it occurs in
    ridge_con_dict = dict() 
    for con in ridge_con:
        ridge_con_dict[con.c1] = con
    
    #the set of ridge constraints that should continue to exist
    ridge_con_s = set(ridge_con)

        
    new_cons = []
    for con in fuse_con:
        lam1 = lamR
        lam2 = lamR
        if con.c1 in ridge_con_dict and ridge_con_dict[con.c1] in ridge_con_s:
            lam1 = ridge_con_dict[con.c1].lam
            ridge_con_s.remove(ridge_con_dict[con.c1])
            
        if con.c2 in ridge_con_dict and ridge_con_dict[con.c2] in ridge_con_s:
            lam2 = ridge_con_dict[con.c2].lam
            ridge_con_s.remove(ridge_con_dict[con.c2])
            

    
            
            
        ridge_con_s.add(constraint(con.c1, None, lam1a))
        ridge_con_s.add(constraint(con.c2, None, lam2a))

    return list(ridge_con_s)

#enumerates constraints from orthology
#args as in solve_ortho_direct
def orth_to_constraints(organisms, gene_ls, tf_ls, orth, lamS):
    #build some maps
    org_to_ind = {organisms[x] : x for x in range(len(organisms))}
    gene_to_inds = map(lambda o: {gene_ls[o][x] : x for x in range(len(gene_ls[o]))}, range(len(organisms)))
    tf_to_inds = map(lambda o: {tf_ls[o][x] : x for x in range(len(tf_ls[o]))}, range(len(organisms)))
    
    #turn orth into a list of all pairs of coefficients
    orth_pairs = reduce(lambda x,y: x+y, map(all_pairs, orth),[])
    
    ortho_dict = dictl()
    for op in orth_pairs:
        (gene1, gene2) = op
        ortho_dict[gene1].append(gene2)
        ortho_dict[gene2].append(gene1)

    
    constraints = []
    for org_i in range(len(organisms)):
        for tf_i in range(len(tf_ls[org_i])):
            for g_i in range(len(gene_ls[org_i])):
                tf = one_gene(tf_ls[org_i][tf_i], organisms[org_i])
                g = one_gene(gene_ls[org_i][g_i], organisms[org_i])
                
                for tf_orth in ortho_dict[tf]:
                    for g_orth in ortho_dict[g]:
                        sub1 = org_i
                        sub2 = org_to_ind[tf_orth.organism]
                        sub3 = org_to_ind[g_orth.organism]
                        
                        if not sub2 == sub3:
                            continue
                        if not tf_orth.name in tf_to_inds[sub2]:
                            continue #if a tf is orthologous to a non-tf
                                              
                        coeff1 = coefficient(sub1, tf_to_inds[sub1][tf.name], gene_to_inds[sub1][g.name])
                        coeff2 = coefficient(sub2, tf_to_inds[sub2][tf_orth.name], gene_to_inds[sub2][g_orth.name])
                        

                        constr = constraint(coeff1, coeff2, lamS)
                        constraints.append(constr)
    
    return constraints
                                             

#gene_ls/tf_ls: lists of gene names and tf names for each problem
#Xs: list of TF expression matrices
#YS: list of gene expression matrices
#Orth: list of lists of one_genes
#priors: list of lists of one_gene pairs
def solve_ortho_direct(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, adjust,lamP, lamR, lamS):
    ridge_con = priors_to_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR)
    #ridge_con = get_ridge_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR, lamR)
    fuse_con = orth_to_constraints(organisms, gene_ls, tf_ls, orth, lamS)
    ridge_con = adjust_ridge_fused(fuse_con, ridge_con, lamR)
    if adjust=='adjust':
        Bs = direct_solve_factor_adjust(Xs, Ys, fuse_con, ridge_con, lamR)
    else:
        Bs = direct_solve_factor(Xs, Ys, fuse_con, ridge_con, lamR)    
    return Bs

def solve_ortho_direct_refit(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS, it, k):
    ridge_con = priors_to_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR)
    #ridge_con = get_ridge_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR, lamR)
    print 'got ridge constraints'
    fuse_con = orth_to_constraints(organisms, gene_ls, tf_ls, orth, lamS)
    print 'got fusion constraints'
    ridge_con = adjust_ridge_fused(fuse_con, ridge_con, lamR)
    print 'adjusted constraints'
    Bs = direct_solve_factor_support(Xs, Ys, fuse_con, ridge_con, lamR)
    print 'solved one'
    
    for i in range(1, it):
        #n = round(2**(it - i - 1) * float(k))        
        #print n
        support = compute_support(Bs, i, it-1, k)
        print 'computed new support'
        Bs = direct_solve_factor_support(Xs, Ys, fuse_con, ridge_con, lamR, support)
        print 'solved another'
    return Bs

#same as above, but without orth_to_constraints. to be used for benchmarking, where orthology is between entries of Bs, not between genes/tfs
def solve_ortho_direct_refit_bench(organisms, gene_ls, tf_ls, Xs, Ys, constraints, priors, lamP, lamR, lamS, it, k):
    ridge_con = priors_to_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR)
    #ridge_con = get_ridge_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR, lamR)
    print 'got ridge constraints'
    fuse_con = constraints
    print 'got fusion constraints'
    ridge_con = adjust_ridge_fused(fuse_con, ridge_con, lamR)
    print 'adjusted constraints'
    Bs = direct_solve_factor_support(Xs, Ys, fuse_con, ridge_con, lamR)
    print 'solved one'
    Bs_0 = Bs
    #print Bs
    for i in range(1, it):
        support = compute_support(Bs, i, it-1, k)
        print 'computed new support'
        #print support
        Bs = direct_solve_factor_support(Xs, Ys, fuse_con, ridge_con, lamR, support)
        print 'solved another'
        #print support[0]
    #plot_thresh_fn(Bs_0, Bs, fuse_con)
    return Bs

def solve_ortho_scad_refit(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS, it, k, s_it):
    ridge_con = priors_to_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR)
    #ridge_con = get_ridge_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR, lamR)
    print 'got ridge constraints'
    fuse_con = orth_to_constraints(organisms, gene_ls, tf_ls, orth, lamS)
    print 'got fusion constraints'
    ridge_con = adjust_ridge_fused(fuse_con, ridge_con, lamR)
    print 'adjusted constraints'
    Bs = solve_scad(Xs, Ys, fuse_con, ridge_con, lamR, lamS, support=None, s_it=s_it)
    print 'solved one'    
    for i in range(1, it):
        n = round(2**(it - i - 1) * float(k))        
        print n
        support = compute_support(Bs, i, it-1, k)
        print 'computed new support'
        Bs = solve_scad(Xs, Ys, fuse_con, ridge_con, lamR, lamS, support, s_it)
        print 'solved another'
    return Bs

def solve_ortho_scad_refit_bench(organisms, gene_ls, tf_ls, Xs, Ys, constraints, priors, lamP, lamR, lamS, it, k, s_it):
    ridge_con = priors_to_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR)
    #ridge_con = get_ridge_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR, lamR)
    print 'got ridge constraints'
    fuse_con = constraints
    print 'got fusion constraints'
    ridge_con = adjust_ridge_fused(fuse_con, ridge_con, lamR)
    print 'adjusted constraints'
    Bs = solve_scad(Xs, Ys, fuse_con, ridge_con, lamR, lamS, support=None, s_it=s_it)
    print 'solved one'
    #plot_scad(Bs, fuse_con)
    #plot_thresh_fn(Bs, Bs, fuse_con)
    Bs_0 = direct_solve_factor(Xs, Ys, [], ridge_con, lamR)
    #randconstraints = []
    #for i in range(1,10000):
    #    r1 = random.randrange(1, Bs[0].shape[0]+1)
    #    c1 = random.randrange(1, Bs[0].shape[1]+1)
    #    r2 = random.randrange(1, Bs[1].shape[0]+1)
    #    c2 = random.randrange(1, Bs[1].shape[1]+1)
    #    randconstr = constraint(coefficient(1, r1, c1), coefficient(2, r2, c2), lamS)
    #randconstraints = scad(Bs, randconstraints, lamS, 3.7)

    #plot_scad(Bs, randconstraints)
    
    for i in range(1, it):

        support = compute_support(Bs, i, it-1, k)
        print 'computed new support'
        Bs = solve_scad(Xs, Ys, fuse_con, ridge_con, lamR, lamS, support, s_it)
        print 'solved another'
    #plot_thresh_fn(Bs_0, Bs, fuse_con)
    return Bs

    
#iteratively adjusts fusion constraint weight to approximate saturating penalty
def solve_scad(Xs, Ys, fuse_con, ridge_con, lamR, lamS, support, s_it):
    fuse_con2 = fuse_con    
    a = 3.7#1.5 #3.7 #!?
    for i in range(s_it):
        Bs = direct_solve_factor_support(Xs, Ys, fuse_con2, ridge_con, lamR, support)
        
        fuse_con2 = scad(Bs, fuse_con, lamS, a)
        #plot_scad(Bs, fuse_con2)
    return Bs

#returns a new set of fusion constraints corresponding to a saturating penalty
def scad(Bs_init, fuse_constraints, lamS, a):
    new_fuse_constraints = []
    for i in range(len(fuse_constraints)):
        con = fuse_constraints[i]
        b_init_1 = Bs_init[con.c1.sub][con.c1.r, con.c1.c]
        b_init_2 = Bs_init[con.c2.sub][con.c2.r, con.c2.c]
        theta_init = np.abs(b_init_1 - b_init_2)
        olamS = con.lam
        if theta_init <= lamS:
            nlamS = lamS
        else:
            
            nlamS = lamS* max(0, ((a*lamS - theta_init)/((a-1)*lamS)))/(2*theta_init)
        new_con = constraint(con.c1, con.c2, nlamS)
        new_fuse_constraints.append(new_con)
    return new_fuse_constraints

def plot_scad(Bs, fuse_constraints):
    from matplotlib import pyplot as plt
    deltab = []
    penalty = []
    for i in range(len(fuse_constraints)):
        con = fuse_constraints[i]
        b_1 = Bs[con.c1.sub][con.c1.r, con.c1.c]
        b_2 = Bs[con.c2.sub][con.c2.r, con.c2.c]
        deltab.append(b_1-b_2)
        penalty.append(con.lam * (b_1-b_2)**2)
    plt.scatter(deltab, penalty)
    plt.xlabel('delta B')
    plt.ylabel('penalty')
    plt.title(con.lam)
    plt.figure()
    #plt.hist(penalty, bins=30)
    plt.show()

def plot_thresh_fn(Bs_init, Bs_current, fuse_constraints):
    from matplotlib import pyplot as plt
    deltab = []
    pen_deltab = []
    for i in range(len(fuse_constraints)):
        con = fuse_constraints[i]
        bi_1 = Bs_init[con.c1.sub][con.c1.r, con.c1.c]
        bi_2 = Bs_init[con.c2.sub][con.c2.r, con.c2.c]
        deltab. append(bi_1-bi_2)
        bc_1 = Bs_current[con.c1.sub][con.c1.r, con.c1.c]
        bc_2 = Bs_current[con.c2.sub][con.c2.r, con.c2.c]
        pen_deltab.append(bc_1-bc_2)
    a=min(deltab)
    b=max(deltab)
    x=np.linspace(a,b,100)
    y=x
    plt.plot(x,y)
    plt.scatter(deltab, pen_deltab)
    plt.xlabel('delta B')
    plt.ylabel('penalized delta B')
    plt.title(con.lam)
    plt.show()


#solves the fused regression problem with constraints coming from groups
#Xs: list of X matrices
#Ys: lsit of Y matrices
#groups: evil mess of a list
#    each entry in groups is a list of 2 elements
#        each entry in the first element is a list of (tf, gene) pairs
#        each entry in the second element is a list of (tf, gene) pairs
#    these (tf, gene) pairs are, collectively, fused to one another
def solve_group_direct(Xs, Ys, groups, priorset, lamP, lamR, lamS):
    fuse_constraints = make_fusion_constraints(groups, lamS)
    ridge_constraints = make_ridge_constraints(priorset, lamP*lamR)
    
    return direct_solve_factor(Xs, Ys, fuse_constraints, ridge_constraints, lamR) 

#solves the fused regression problem with constraints coming from groups
#Xs: list of X matrices
#Ys: lsit of Y matrices
#groups: evil mess of a list
#    each entry in groups is a list of 2 elements
#        each entry in the first element is a list of (tf, gene) pairs
#        each entry in the second element is a list of (tf, gene) pairs
#    these (tf, gene) pairs are, collectively, fused to one another
def solve_group_iter(Xs, Ys, groups, priorset, lamP, lamR, lamS,it):
    fuse_constraints = make_fusion_constraints(groups, lamS)
    ridge_constraints = [] # TODO
    return iter_solve(Xs, Ys, fuse_constraints, ridge_constraints, lamR, it) 
     

def make_ridge_constraints(ridge, lam):
    constraints = []
    for sub in range(len(ridge)):
        ridgecons = ridge[sub]
        for ridgecon in ridgecons:
            (r, c) = ridgecon
            con = constraint(coefficient(sub, r, c), None, lam)
            constraints.append(con)
    return constraints
#makes constraints from the list of fusion groups given by groups
#groups: [ [ sub1list, sub2list], ... ]
#sub1list/sub2list: [(tf, gene), ...]
#enumerates every fusion constraint between a (tf, gene) in sublist 1 and a (tf, gene) in sublist2
#IMPORTANT: this only adds sub1 -> sub2 constraints
#this is because constraints are treated as symmetric
def make_fusion_constraints(groups, lam):
    Cs=[]
    
    for group in range(len(groups)):
        
        sub1 = groups[group][0]
        
        sub2 = groups[group][1]
        for coeff1 in sub1:
            for coeff2 in sub2:
                
                (tf1, g1) = coeff1
                (tf2, g2) = coeff2
                Cs.append(constraint(coefficient(0, tf1, g1), coefficient(1, tf2, g2), lam))
                
    return Cs

#solves W = argmin_W ((XW - Y)**2).sum() + constraint related terms
#Xs, Ys: X and Y for each subproblem
#fuse_constraints: fusion constraints
#ridge_constraints: ridge regression constraints. constraints not mentioned are assumed to exist with lam=lambdaR
#it: number of iterations to run
def iter_solve(Xs, Ys, fuse_constraints, ridge_constraints, lambdaR, it):
    Bs = []
    #set the initial guess (all zero)
    for j in range(len(Xs)):
        Bs.append(np.zeros((Xs[j].shape[1], Ys[j].shape[1])))
    Bss = [Bs, map(lambda x: x.copy(), Bs)]
        
    for i in range(it):
        cB = Bss[np.mod(i, 2)] #current B and other B
        oB = Bss[np.mod(i+1, 2)]
        lam_ramp = (1.0+i)/it #constant to multiply by lam
        for s in range(len(Xs)):
            X = Xs[s] #X, Y for current subproblem s
            Y = Ys[s]
            for c in range(X.shape[1]): #column of B we are solving
                y = Y[:, [c]]
                I = np.eye(X.shape[1])*lambdaR*lam_ramp
                ypad_l = []
                xpad_l = []
                
                
                #add in the fusion constraints
                for con in fuse_constraints:
                    if con.c1.c == c and con.c1.sub == s:
                        targ = oB[s][con.c2.r,con.c2.c]*con.lam*lam_ramp
                        xpad_l.append(np.zeros((1,X.shape[1])))
                        ypad_l.append(targ)
                        xpad_l[-1][0, con.c1.r] = con.lam*lam_ramp
                        xpad_l[-1][0, con.c2.r] = -con.lam*lam_ramp
                if len(xpad_l):
                    xpad = np.vstack(xpad_l)
                else:
                    xpad = np.zeros((0, X.shape[1]))
                #now we add in zeros for the ridge
                ypad_l.append(np.zeros((X.shape[1], 1)))
                ypad = np.vstack(ypad_l)
                #add in the ridge constraints
                for con in ridge_constraints:
                    if con.c1.c == c and con.c1.sub == s:
                        I[con.r, con.r] = con.lam*lam_ramp
                    
                F = np.vstack((X, xpad, I))
                p = np.vstack((y, ypad))
                #we solve b that minimizes Fb = p
                
                (b, resid, rank, sing) = np.linalg.lstsq(F, p)
                
                cB[j][:, [c]] = b
    return Bss[np.mod(it,2)]


#diagonally concatenates two matrices
#TODO: sparse
def diag_concat(mats):

    tot_rows=sum(map(lambda x: x.shape[0], mats))
    tot_cols=sum(map(lambda x: x.shape[1], mats))
    A=np.zeros((tot_rows, tot_cols))
    row=0
    col=0
    for i in range(0,len(mats)):
        mi = mats[i]
        A[row:(row+mi.shape[0]), col:(col+mi.shape[1])] = mi
        row += mi.shape[0]
        col += mi.shape[1]
    return A

#this needs commenting!
def direct_solve_factor(Xs, Ys, fuse_constraints, ridge_constraints, lambdaR):
    #(coeff_l, con_l) = factor_constraints(Xs, Ys, fuse_constraints)
    (coeff_l, con_l) = factor_constraints_columns(Xs, Ys, fuse_constraints)
    
    Bs = []
        
    #Initialize matrices to hold solutions
    for i in range(len(Xs)):
        Bs.append(np.zeros((Xs[i].shape[1], Ys[i].shape[1])))
    print 'starting solver'
    #iterate over constraint sets
    for f in range(len(coeff_l)):
        print('\r working on subproblem: %d'%f), #!?!?!?!
        #get the coefficients and constraints associated with the current problem
        coefficients = coeff_l[f]
        constraints = con_l[f]
        ridge_cons = []
        
        coefficients_set = set(coefficients)

        for con in ridge_constraints:
            coeff = con.c1
            if coeff in coefficients_set:
                ridge_cons.append(con)
    
        columns = set()

        for co in coefficients:
            columns.add(coefficient(co.sub, None, co.c))
        columns = list(columns)
        
        (constraints, ridge_cons) = adjust_vol2(Xs, columns, constraints, ridge_cons,lambdaR)
        
        num_subproblems = len(set(map(lambda co: co.sub, coefficients)))
        
        #we're building a canonical ordering over the columns for the current set of columns
        counter = 0
        sub_map = dict()
        for co in columns:
            sub = co.sub
            if not (co.sub, co.c) in sub_map:
                sub_map[(sub, co.c)] = counter
                counter += 1
                
        #make X
        X_l = []
        for co in columns:
            X_l.append(Xs[co.sub])
        #make Y. 
        Y_l = []
        for co in columns:
            Y = Ys[co.sub]
            Y_l.append(Y[:, [co.c]])

        #compute a cumulative sum over the number of columns in each sub-block, for use as offsets when computing coefficient indices for penalties

        cums = [0]+list(np.cumsum(map(lambda x: x.shape[1], X_l)))
        P_l = [np.zeros((0, cums[-1]))]
                
        for con in constraints:
            P = np.zeros((1, cums[-1]))
            #get the indices of the diagonal blocks in X corresponding to the coefficients in this constraint
            ind1 = sub_map[(con.c1.sub, con.c1.c)]
            ind2 = sub_map[(con.c2.sub, con.c2.c)]
            #the locations corresponding to the coefficients we want to penalize are the start index of that block plus the row
            pc1 = cums[ind1] + con.c1.r
            pc2 = cums[ind2] + con.c2.r
            P[0, pc1] = con.lam
            P[0, pc2] = -con.lam
            
            P_l.append(P)
        #set up ridge constraint. 
        I = np.eye((cums[-1])) * lambdaR
        
        #now we go through the ridge constraints and set entries of I
        for con in ridge_constraints:
            coeff = con.c1
            if coeff in coefficients_set:
                ind1 = sub_map[(con.c1.sub, con.c1.c)]
                pc1 = cums[ind1] + con.c1.r
                I[pc1, pc1] = con.lam
        


        #now add an appropriate number of zeros to Y_l
        Y_l.append(np.zeros((cums[-1] + len(constraints), 1)))
        
        P = np.vstack(P_l)
        X = np.vstack((diag_concat(X_l), P, I))
        y = np.vstack(Y_l)
        
        Xsp = scipy.sparse.csr_matrix(X)
        
        ysp = scipy.sparse.csr_matrix(y)
        bsp = scipy.sparse.linalg.lsqr(Xsp, y)#returns many things!
        b = bsp[0][:, None] #god this is annoying


        #(b, resid, rank, sing) = np.linalg.lstsq(X, y)        
        
        #now we put it all together
        for co_i in range(len(columns)):
            co = columns[co_i]
            start_ind = cums[co_i]
            end_ind = cums[co_i+1]
            
            Bs[co.sub][:, [co.c]] = b[start_ind:end_ind]
        
        
    return Bs

def direct_solve_factor_adj(Xs, Ys, fuse_constraints, ridge_constraints, lambdaR):
    #(coeff_l, con_l) = factor_constraints(Xs, Ys, fuse_constraints)
    (coeff_l, con_l) = factor_constraints_columns(Xs, Ys, fuse_constraints)
    
    Bs = []
        
    #Initialize matrices to hold solutions
    for i in range(len(Xs)):
        Bs.append(np.zeros((Xs[i].shape[1], Ys[i].shape[1])))
    print 'starting solver'
    #iterate over constraint sets
    for f in range(len(coeff_l)):
        print('\r working on subproblem: %d'%f), #!?!?!?!
        #get the coefficients and constraints associated with the current problem
        coefficients = coeff_l[f]
        constraints = con_l[f]
        ridge_cons = []
        
        coefficients_set = set(coefficients)

        for con in ridge_constraints:
            coeff = con.c1
            if coeff in coefficients_set:
                ridge_cons.append(con)
    
        columns = set()

        for co in coefficients:
            columns.add(coefficient(co.sub, None, co.c))
        columns = list(columns)
        
        (constraints, ridge_cons) = adjust_vol2(Xs, columns, constraints, ridge_cons,lambdaR)
        
        num_subproblems = len(set(map(lambda co: co.sub, coefficients)))
        
        #we're building a canonical ordering over the columns for the current set of columns
        counter = 0
        sub_map = dict()
        for co in columns:
            sub = co.sub
            if not (co.sub, co.c) in sub_map:
                sub_map[(sub, co.c)] = counter
                counter += 1
                
        #make X
        X_l = []
        for co in columns:
            X_l.append(Xs[co.sub])
        #make Y. 
        Y_l = []
        for co in columns:
            Y = Ys[co.sub]
            Y_l.append(Y[:, [co.c]])

        #compute a cumulative sum over the number of columns in each sub-block, for use as offsets when computing coefficient indices for penalties

        cums = [0]+list(np.cumsum(map(lambda x: x.shape[1], X_l)))
        P_l = [np.zeros((0, cums[-1]))]
                
        for con in constraints:
            P = np.zeros((1, cums[-1]))
            #get the indices of the diagonal blocks in X corresponding to the coefficients in this constraint
            ind1 = sub_map[(con.c1.sub, con.c1.c)]
            ind2 = sub_map[(con.c2.sub, con.c2.c)]
            #the locations corresponding to the coefficients we want to penalize are the start index of that block plus the row
            pc1 = cums[ind1] + con.c1.r
            pc2 = cums[ind2] + con.c2.r
            P[0, pc1] = con.lam
            P[0, pc2] = -con.lam
            
            P_l.append(P)
        #set up ridge constraint. 
        I = np.eye((cums[-1])) * lambdaR
        
        #now we go through the ridge constraints and set entries of I
        for con in ridge_constraints:
            coeff = con.c1
            if coeff in coefficients_set:
                ind1 = sub_map[(con.c1.sub, con.c1.c)]
                pc1 = cums[ind1] + con.c1.r
                I[pc1, pc1] = con.lam
        


        #now add an appropriate number of zeros to Y_l
        Y_l.append(np.zeros((cums[-1] + len(constraints), 1)))
        
        P = np.vstack(P_l)
        X = np.vstack((diag_concat(X_l), P, I))
        y = np.vstack(Y_l)
        
        Xsp = scipy.sparse.csr_matrix(X)
        
        ysp = scipy.sparse.csr_matrix(y)
        bsp = scipy.sparse.linalg.lsqr(Xsp, y)#returns many things!
        b = bsp[0][:, None] #god this is annoying


        #(b, resid, rank, sing) = np.linalg.lstsq(X, y)        
        
        #now we put it all together
        for co_i in range(len(columns)):
            co = columns[co_i]
            start_ind = cums[co_i]
            end_ind = cums[co_i+1]
            
            Bs[co.sub][:, [co.c]] = b[start_ind:end_ind]
        
        
    return Bs

#return a new support, consisting of the top k regressors for each col
def compute_support(Bs, i, it, n):
    Bs_k = []
    for B in Bs:
        #Bs.shape[0] * p**it = n
        #(k/Bs.shape[0])**(1/it) = p
        p = (float(n)/B.shape[0])**(1.0/it)
        k = (p**i) * B.shape[0]
        Bi = np.argsort(-np.abs(B), axis=0)
        Bk = np.zeros(B.shape)
        for c in range(Bk.shape[1]):

            Bk[Bi[0:k, c], c] = 1
        Bs_k.append(Bk)
    return Bs_k

#just like direct_solve_factor, but also takes a list of matrixes (same dims as B) specifying the support.
def direct_solve_factor_support(Xs, Ys, fuse_constraints, ridge_constraints, lambdaR, support=None):
    if support == None:
        support = []
        for i in range(len(Xs)):
            support.append(np.ones((Xs[i].shape[1], Ys[i].shape[1])))
    #first thing we do is remove unsupported fusion constraints
    def constraint_supported(con):
        supported = True
        for co in (con.c1, con.c2):
            supported = supported and support[co.sub][co.r, co.c]
        return supported
    fuse_constraints = filter(constraint_supported, fuse_constraints)


    #(coeff_l, con_l) = factor_constraints(Xs, Ys, fuse_constraints)
    (coeff_l, con_l) = factor_constraints_columns(Xs, Ys, fuse_constraints)
    Bs = []
        
    #Initialize matrices to hold solutions
    for i in range(len(Xs)):
        Bs.append(np.zeros((Xs[i].shape[1], Ys[i].shape[1])))
    
    #iterate over constraint sets
    for f in range(len(coeff_l)):
        
        print('\r working on subproblem: %d'%f), #!?!?!?!
        #get the coefficients and constraints associated with the current problem
        coefficients = coeff_l[f]
        constraints = con_l[f]
        
        columns = set()

        for co in coefficients:
            columns.add(coefficient(co.sub, None, co.c))
        columns = list(columns)
        
        num_subproblems = len(set(map(lambda co: co.sub, coefficients)))
        
        #we're building a canonical ordering over the columns for the current set of columns
        counter = 0
        sub_map = dict()
        for co in columns:
            sub = co.sub
            if not (co.sub, co.c) in sub_map:
                sub_map[(sub, co.c)] = counter
                counter += 1
                
        #make X
        X_l = []
        for co in columns:
            #get the support for the current subproblem/column of outp
            col_support = support[co.sub][:, co.c]
            Xc = Xs[co.sub] * col_support #multiplies columns by 0/1
            X_l.append(Xc)
        #make Y. 
        Y_l = []
        for co in columns:
            Y = Ys[co.sub]
            Y_l.append(Y[:, [co.c]])

        #compute a cumulative sum over the number of columns in each sub-block, for use as offsets when computing coefficient indices for penalties

        cums = [0]+list(np.cumsum(map(lambda x: x.shape[1], X_l)))
        P_l = [np.zeros((0, cums[-1]))]
                
        for con in constraints:
            P = np.zeros((1, cums[-1]))
            #get the indices of the diagonal blocks in X corresponding to the coefficients in this constraint
            ind1 = sub_map[(con.c1.sub, con.c1.c)]
            ind2 = sub_map[(con.c2.sub, con.c2.c)]
            #the locations corresponding to the coefficients we want to penalize are the start index of that block plus the row
            pc1 = cums[ind1] + con.c1.r
            pc2 = cums[ind2] + con.c2.r
            P[0, pc1] = con.lam
            P[0, pc2] = -con.lam
            
            P_l.append(P)
        #set up ridge constraint. 
        I = np.eye((cums[-1])) * lambdaR
        coefficients_set = set(coefficients)
        #now we go through the ridge constraints and set entries of I
        for con in ridge_constraints:
            coeff = con.c1
            if coeff in coefficients_set:
                ind1 = sub_map[(con.c1.sub, con.c1.c)]
                pc1 = cums[ind1] + con.c1.r
                I[pc1, pc1] = con.lam
        


        #now add an appropriate number of zeros to Y_l
        Y_l.append(np.zeros((cums[-1] + len(constraints), 1)))
        
        P = np.vstack(P_l)
        X = np.vstack((diag_concat(X_l), P, I))
        y = np.vstack(Y_l)
        t1 = time.time()
        Xsp = scipy.sparse.csr_matrix(X)
        
        ysp = scipy.sparse.csr_matrix(y)
        bsp = scipy.sparse.linalg.lsqr(Xsp, y)#returns many things!
        b = bsp[0][:, None] #god this is annoying
        
        #now we put it all together
        for co_i in range(len(columns)):
            co = columns[co_i]
            start_ind = cums[co_i]
            end_ind = cums[co_i+1]
            
            Bs[co.sub][:, [co.c]] = b[start_ind:end_ind]
        
        
    return Bs


        
from matplotlib import pyplot as plt
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
        

#no cleverness at all
#this is here as a sanity check
def direct_solve(Xs, Ys, fuse_constraints, ridge_constraints, lambdaR, it):
    ncols = sum(map(lambda y: y.shape[1], Ys))
    nrowsX = sum(map(lambda x: x.shape[0], Xs))
    ncolsX = sum(map(lambda x: x.shape[1], Xs))
    X = diag_concat(Xs * ncols)
    y = diag_concat(Ys).ravel(order='F')[:, np.newaxis]
    #y = np.vstack(map(lambda x: x.ravel()[:, np.newaxis], Ys)*ncols)
    
    #finds the flattened index in b of subproblem sub, row r, col c
    def indb(sub, r, c):
        ind = ncols * c + r
    xpad_l = [X]

    ypad_l = [y]
    I = np.eye(X.shape[1])*lambdaR
    for con in ridge_constraints:
        ind = indb(con.c1.sub, con.c1.r, conc1.c)
        I[ind, ind] = con.lam
    for con in fuse_constraints:
        xpad_l.append(np.zeros((1, X.shape[1])))
        ypad_l.append(0)
        ind1 = indb(con.c1.sub, con.c1.r, con.c1.c)
        ind2 = indb(con.c2.sub, con.c2.r, con.c2.c)
        xpad_l[-1][0, ind1] = con.lam
        xpad_l[-1][0, ind2] = -con.lam
    xpad_l.append(I)
    ypad_l.append(np.zeros((X.shape[1], 1)))
    
    F = np.vstack(xpad_l)
    p = np.vstack(ypad_l)
    
    (b, resid, rank, sing) = np.linalg.lstsq(F, p)    

    B = np.reshape(b, (ncolsX, ncols),order='F')

        #now collect Bs
    Bs = []
    crow = 0
    ccol = 0
    
    for i in range(len(Xs)):
        extr = Xs[i].shape[1]
        extc = Ys[i].shape[1]
        Bs.append(B[crow:(crow + extr), ccol:(ccol+extc)])
    return Bs


#TODO: test this
def factor_constraints_columns(Xs, Ys, constraints):
    columns = []
    for sub in range(len(Xs)):
        for c in range(Ys[sub].shape[1]):
            columns.append((sub, c))
            
    constraints_c = set()
    for con in constraints:
        constraints_c.add(constraint(coefficient(con.c1.sub, None, con.c1.c), coefficient(con.c2.sub, None, con.c2.c), None))
        constraints_c.add(constraint( coefficient(con.c2.sub, None, con.c2.c), coefficient(con.c1.sub, None, con.c1.c), None))

    
    not_visited = set(columns)
    
    def factor_helper(col_fr, col_l):
        for col_to in columns:
            if not col_to in not_visited:
                continue
            potential_con = constraint(coefficient(col_fr[0], None, col_fr[1]), coefficient(col_to[0], None, col_to[1]), None)
            
            
            if potential_con in constraints_c:
                col_l.append(col_to)
                not_visited.remove(col_to)
                factor_helper(col_to, col_l)
    col_subls = []
    while len(not_visited):
        
        col_fr = not_visited.pop()
        
        col_l = [col_fr]
        factor_helper(col_fr, col_l)
        col_subls.append(col_l)
        
        
    coeffs_l = []
    cons_l = []

    coeff_to_constraints = dict()
    for con in constraints:
        if con.c1 in coeff_to_constraints:
            coeff_to_constraints[con.c1].append(con)
        else:
            coeff_to_constraints[con.c1] = [con]
        if con.c2 in coeff_to_constraints:
            coeff_to_constraints[con.c2].append(con)
        else:
            coeff_to_constraints[con.c2] = [con]
            


    for col_subl in col_subls:
        #now we need to get all the coefficients that really go here
        coeffs = []
        cons = []
        for col in col_subl:
            (sub, c) = col
            for r in range(Xs[sub].shape[1]):
                coeff = coefficient(sub, r, c)
                coeffs.append(coeff)
                if coeff in coeff_to_constraints:
                    for con in coeff_to_constraints[coeff]:    
                        cons.append(con)
        coeffs_l.append(coeffs)
        cons_l.append(cons)
    print '\ndone: %d subproblems, %d columns'%(len(coeffs_l), len(columns))
    return (coeffs_l, cons_l)

#factors the constraints!
#this is a bit slower than it really needs to be because it isn't really working based on columns
def factor_constraints(Xs, Ys, constraints):
    cset = set() 
    cmap = dict()
    #enumerate the coefficients
    for sub in range(len(Xs)):
        
        for r in range(Xs[sub].shape[1]):
            for c in range(Ys[sub].shape[1]):
                coeff = coefficient(sub, r, c)
                cset.add(coeff)
                cmap[coeff] = []
                
    #build a coefficient -> [constraints, ...] map
    for con in constraints:
        cmap[con.c1].append(con)
        cmap[con.c2].append(con) 
        
    #starts with the last entry in coeffs and recursively adds connected constraints and coefficients to the lists cons and coeffs, while removing coefficients from cset
    #every constraint connected to the current coefficient is added
    #recursively called on coefficients linked by constraints or in the same column, which have not already been visited

    #cset: set containing unvisited constraints
    #cons: list of constraints 
    #coeffs: list of coefficients in current problem
    def factor_constraints_helper(cset, cons, coeffs):
        coe_fr = coeffs[-1]
        #first go down the column
        for r in range(Xs[coe_fr.sub].shape[1]):
            col_nbr = coefficient(coe_fr.sub, r, coe_fr.c)
            if col_nbr in cset:
                cset.remove(col_nbr)
                coeffs.append(col_nbr)
                factor_constraints_helper(cset, cons, coeffs)
        #now go down the constraints
        for con in cmap[coe_fr]:
            #we traverse edges in both directions, but only keep track of the ones that have the same direction we're traveling
            if con.c1 == coe_fr:
                cons.append(con)
            for coe_to in (con.c1, con.c2):
                if coe_to in cset:
                    coeffs.append(coe_to)
                    cset.remove(coe_to)
                    factor_constraints_helper(cset, cons, coeffs)        
    coeff_l = []
    con_l = []
    while len(cset):
        coeff = cset.pop()
        print 'factoring starting with '+str(coeff)
        print str(len(cset)) + ' left'
        coeff_l.append([coeff])
        con_l.append([])
        factor_constraints_helper(cset, con_l[-1], coeff_l[-1])
    return (coeff_l, con_l)
