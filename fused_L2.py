import numpy as np
import collections
import time
coefficient = collections.namedtuple('coefficient',['sub','r','c'])

#c2 == None is a constant constraint
constraint = collections.namedtuple('constraint',['c1','c2','lam'])


one_gene = collections.namedtuple('gene',['name','organism'])

#returns a list of lists of all pairs of entries from l, where the first entry in the pair occurs before the second in l
def all_pairs(l):
    pl = []
    for li1 in range(len(l)):
        for li2 in range(li1+1, len(l)):
           pl.append([li1, li2])
    return pl

class dictl(dict):
    def __getitem__(self, i):
        if not i in self:
            self[i] = []
        return super(dictl, self).__getitem__(i)

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
        ortho_dict[gene2].append(gene2)
    
    constraints = []
    for org_i in range(len(organisms)):
        for tf_i in range(len(tf_ls[org_i])):
            for g_i in range(len(gene_ls[org_i])):
                tf = one_gene(tf_ls[org_i][tf_i], organisms[org_i])
                g = one_gene(gene_ls[org_i][g_i], organisms[org_i])
                for tf_orth in ortho_dict[tf]:
                    for g_orth in ortho-dict[g]:
                        sub1 = org_i
                        sub2 = org_to_ind[tf_orth.organism]
                        sub3 = org_to_ind[g_orth.organism]
                        if not sub2 == sub3:
                            continue
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
def solve_ortho_direct(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS):
    ridge_con = priors_to_constraints(organisms, gene_ls, tf_ls, priors, lamP*lamR)
    fuse_con = orth_to_constraints(organisms, gene_ls, tf_ls, orth, lamS)
    Bs = direct_solve_factor(Xs, Ys, fuse_con, ridge_con, lamR)
    return Bs


    
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
    ridge_constraints = [] # TODO
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
    
    #iterate over constraint sets
    for f in range(len(coeff_l)):
        print 'working on subproblem %d of %d'%(f,len(coeff_l))
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
        #set up ridge constraint. TODO: implement priors
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
        
        (b, resid, rank, sing) = np.linalg.lstsq(X, y)        
        
        #now we put it all together
        for co_i in range(len(columns)):
            co = columns[co_i]
            start_ind = cums[co_i]
            end_ind = cums[co_i+1]
            
            Bs[co.sub][:, [co.c]] = b[start_ind:end_ind]
        
        
    return Bs
        
#from matplotlib import pyplot as plt
def prediction_error(X, B, Y, metric):
    Ypred = np.dot(X, B)
    
    
    #plt.plot(yp[1:200])
    #plt.hold(True)
    #plt.plot(y[1:200])
    #plt.show()
    if metric == 'R2':
        r2a = 0.0
        for c in range(Ypred.shape[1]):
            y = Y[:, c]
            yp = Ypred[:, c]
            r2 = 1 - ((y-yp)**2).sum()/ (y**2).sum()
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
        con_c.add(constraint(coefficient(con.c1.sub, None, con.c1.c), coefficient(con.c2.sub, None, con.c2.c), None))
    
    columns_s = set(columns)
    def factor_helper(col_fr, col_l):
        for col_to in columns_s:
            if constraint(coefficient(col_fr[0], None, col_fr[1]), coefficient(col_to[1], None, col_to[1]), None) in constraints_c:
                col_l.append(col_to)
                columns_s.remove(col_to)
                factor_helper(col_to, col_l)
    col_subls = []
    while len(columns_s):

        col_fr = columns_s.pop()
        print 'factoring starting with '+str(col_fr)
        print str(len(columns_s)) + ' left'

        col_l = [col_fr]
        factor_helper(col_fr, col_l)
        col_subls.append(col_l)
    
    coeffs_l = []
    cons_l = []
    for col_subl in col_subls:
        #now we need to get all the coefficients that really go here
        coeffs = []
        cons = []
        for col in col_subl:
            (sub, c) = col
            for r in range(Xs[sub].shape[1]):
                coeffs.append(coefficient(sub, r, c))
            for con in constraints:
                
                if (con.c1.sub == sub and con.c1.c == c) or (con.c1.sub == sub and con.c1.c == c):
                    cons.append(con)
        coeffs_l.append(coeffs)
        cons_l.append(cons)
    
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
