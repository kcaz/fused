import numpy as np
from numpy import linalg
import random
import collections
import time
import scipy.sparse
import scipy.sparse.linalg


#SECTION: -------------------DATA STRUCTURES--------------
coefficient = collections.namedtuple('coefficient',['sub','r','c'])

#c2 == None is a constant constraint
constraint = collections.namedtuple('constraint',['c1','c2','lam'])

one_gene = collections.namedtuple('gene',['name','organism'])
class dictl(dict):
    def __getitem__(self, i):
        if not i in self:
            self[i] = []
        return super(dictl, self).__getitem__(i)

#SECTION: ------------------------UTILITY FUNCTIONS-------
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

#returns a list of lists of all pairs of entries from l, where the first entry in the pair occurs before the second in l
def all_pairs(l):
    pl = []
    for li1 in range(len(l)):
        for li2 in range(li1+1, len(l)):
           pl.append([l[li1], l[li2]])
    return pl

#solves a quadratic equation, returning both solutions
def quad(a, b, c):
    x1 = 0.5*(-b + (b**2 - 4*a*c)**0.5)/a
    x2 = 0.5*(-b - (b**2 - 4*a*c)**0.5)/a
    
    return (x1, x2)


#SECTION: --------CODE FOR GENERATING CONSTRAINTS---------

#enumerates fusion constraints from orthology
#args as in solve_ortho_direct
#returns list of constraints. individual constraints are pairs of coefficients, with an associated weight lamS
#NOTE!!!! THIS IS FUCKED UP IF THEY HAVE OVERLAPPING GENE NAMES! FIX FIX FIX FIX FIX
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

#given priors, returns a list of constraints. constraints representing priors have their second coefficient equal to None.
#NOTE: this function does not allow priors to have different weights.
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

#adds a "prior" on transcription factors not self-regulating. This is an attempt to avoid degenerate learning of an identity matrix for that part of B
def no_self_reg_constraints(organisms, gene_ls, tf_ls, lam):
    constraints = []
    gene_to_inds = map(lambda o: {gene_ls[o][x] : x for x in range(len(gene_ls[o]))}, range(len(organisms)))
    tf_to_inds = map(lambda o: {tf_ls[o][x] : x for x in range(len(tf_ls[o]))}, range(len(organisms)))

    for subi, org in enumerate(organisms):
        for tfi, tf in enumerate(tf_ls[subi]):
            gi = gene_to_inds[subi][tf]
            constr = constraint(coefficient(subi, tfi, gi), None, lam)
            constraints.append(constr)
    return constraints

#SECTION: --------CODE FOR ADJUSTING CONSTRAINTS---------

#MISSING: adjuster based on variance, because that's not always possible

#we don't want the presence of a fusion constraint to over-constrain the coefficients that the fusion constraint is attached to. This code adjusts the weight of fusion and ridge constraints in order to maintain the determinant of the covariance matrix.
#because default constraints are not represented individually, this code involves introducing new ones sometimes.
#NOTE: I think there are some numerical issues here but I don't remember what they are
#NOTE: not using this right now
def adjust_vol(Xs, cols, fuse_cons, ridge_cons, lamR):
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

    #think about left multiplying by an identity matrix with some constant c in entries that show up in adj_rows in order to equate determinants
    c = (d2/d1)**(1.0/len(adj_rows))

    #this is some code for testing
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



#SECTION: ----------------CODE FOR SOLVING THE MODEL-------------------

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





#most basic solver. Can involve a pre-adjustment step to avoid over-regularizing fused constraints. Solves by factoring.
#gene_ls/tf_ls: lists of gene names and tf names for each problem
#Xs: list of TF expression matrices
#YS: list of gene expression matrices
#Orth: list of lists of one_genes
#priors: list of lists of one_gene pairs
def solve_ortho_direct(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors,lamP, lamR, lamS, adjust=False, self_reg_pen = 0):
    
    
    ridge_con = priors_to_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR)
    fuse_con = orth_to_constraints(organisms, gene_ls, tf_ls, orth, lamS)
    if self_reg_pen:
        self_con = no_self_reg_constraints(organisms, gene_ls, tf_ls, lamR * self_reg_pen)

    Bs = direct_solve_factor(Xs, Ys, fuse_con, ridge_con, lamR)    
    return Bs

#parameters as solve_ortho_direct. 
#s_it defines the number of scad-like iterations to do
def solve_ortho_direct_scad(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS, s_it):
    ridge_con = priors_to_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR)
    fuse_con = orth_to_constraints(organisms, gene_ls, tf_ls, orth, lamS)
    Bs = solve_scad(Xs, Ys, fuse_con, ridge_con, lamR, lamS, s_it=s_it)
    return Bs

#this is like direct solve, but it brakes up unrelated columns
#solves W = argmin_W ((XW - Y)**2).sum() + constraint related terms
#Xs, Ys: X and Y for each subproblem
#fuse_constraints: fusion constraints
#ridge_constraints: ridge regression constraints. constraints not mentioned are assumed to exist with lam=lambdaR
#it: number of iterations to run
def direct_solve_factor(Xs, Ys, fuse_constraints, ridge_constraints, lambdaR, adjust=False):
    
    (coeff_l, con_l) = factor_constraints_columns(Xs, Ys, fuse_constraints)
    
    Bs = []
    
    #Initialize matrices to hold solutions
    for i in range(len(Xs)):
        Bs.append(np.zeros((Xs[i].shape[1], Ys[i].shape[1])))
    #print 'starting solver'
    #iterate over constraint sets
    for f in range(len(coeff_l)):
        #print('\r working on subproblem: %d'%f), #!?!?!?!
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
        
        if adjust:
            print 'haha not implmneted'
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

#iteratively adjusts fusion constraint weight to approximate saturating penalty
def solve_scad(Xs, Ys, fuse_con, ridge_con, lamR, lamS, s_it):
    fuse_con2 = fuse_con    
    a = 3.7#1.5 #3.7 #!?
    for i in range(s_it):
        Bs = direct_solve_factor(Xs, Ys, fuse_con2, ridge_con, lamR)
        
        fuse_con2 = scad(Bs, fuse_con, lamS, a)
        #plot_scad(Bs, fuse_con2)
    return Bs


#solves W = argmin_W ((XW - Y)**2).sum() + constraint related terms iteratively
#Xs, Ys: X and Y for each subproblem
#fuse_constraints: fusion constraints
#ridge_constraints: ridge regression constraints. constraints not mentioned are assumed to exist with lam=lambdaR
#it: number of iterations to run
def iter_solve(Xs, Ys, fuse_constraints, ridge_constraints, lambdaR, it):
    print 'nope'
#this code cuts up columns by depth first search
#returns a list of lists of coefficients associated with each subproblem
#and a list of lists of constraints associated with each subproblem
#NOTE: this is pretty horribly written and should be redone at some point
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
    #print '\ndone: %d subproblems, %d columns'%(len(coeffs_l), len(columns))
    return (coeffs_l, cons_l)