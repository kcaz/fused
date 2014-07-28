import numpy as np
import collections
import time
coefficient = collections.namedtuple('coefficient',['sub','r','c'])

#c2 == None is a constant constraint
constraint = collections.namedtuple('constraint',['c1','c2','lam'])


coeff1 = coefficient(0, 12, 13)

#returns a list mapping c=1...C to constraints with c in the first position
#constraints are directional!!!!!! this makes our life simpler
#but it also means that breaking the symmetry breaks things
def index_constraints(constraints, s, C):
    cmap = []
    for c in range(C):
        cmap.append([])
        for con in constraints:
            if con.sub == s and con.c1.c == c:#hrm direction?
                cmap[-1].append(con)
    return cmap
    
def solve_from_groups_i(Xs, TFs, Ys, Gs, groups, priorset, lamP, lamR, lamS, it):
    fuse_constraints = make_fusion_constraints(groups, lamS)
    ridge_constraints = [] #TODO
    
    return direct_solve_factor(Xs, Ys, fuse_constraints, ridge_constraints, lamR) 
    #return iter_solve(Xs, Ys, fuse_constraints, ridge_constraints, lamR, it)

def solve_group_orth(Xs, Ys, groups, priorset, lamP, lamR, lamS):
    fuse_constraints = make_fusion_constraints(groups, lamS)
    ridge_constraints = [] # TODO
    return direct_solve_factor(Xs, Ys, fuse_constraints, ridge_constraints, lamR) 
    

def pred_err_grps(B, X_lo, Y_lo):
    errs = np.zeros(Y_lo.shape[1])
    predY = np.array(np.dot(X_lo, B))
    for c in range(Y_lo.shape[1]):
        err = np.mean((predY[:, c] - Y_lo[:,c])**2)
        errs[c] = err    
    return errs.mean()

 

#makes constraints from the list of fusion groups given by groups
#groups: [ [ sub1list, sub2list], ... ]
#sub1list/sub2list: [(tf, gene), ...]
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
                #Cs.append(constraint(coefficient(0, tf2, g2), coefficient(1, tf1, g1), lam)) #symmetry important here
                
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

def con_to_inds(TFs, Gs, Cs):
    tfi = map(lambda x: {x[y] : y for y in range(len(x))}, TFs)
    gsi = map(lambda x: {x[y] : y for y in range(len(x))}, Gs)
    
    Cs2 = []
    for con in Cs:
        (i1, tf1, g1, i2, tf2, g2) = con
        Cs2.append((i1, tfi[i1][tf1], gsi[i1][g1], i2, tfi[i2][tf2], gsi[i2][g2]))
    return Cs2

def make_bpad(Bs, j, c, Csi):
    padS = []
    P = []
    #reminder format for Csi: (i, tf1i, g1i, j, tf2i, g2i)
    for con in Csi:
        (i1, t1i, g1i, i2, t2i, g2i) = con
        #print (i1, t1i, g1i, i2, t2i, g2i)
        if i1 == j and g1i == c:
            g2im = np.mod(g2i, Bs[i2].shape[1])
            padS.append(Bs[i2][t2i, g2im])
            Prow = np.zeros((1, Bs[i1].shape[0]))
            Prow[0, t1i] = 1
            P.append(Prow)
        if i2 == j and g2i == c:
            g1im = np.mod(g1i, Bs[i1].shape[1])
            padS.append(Bs[i1][t1i, g1im])
            Prow = np.zeros((1, Bs[i2].shape[0]))
            Prow[0, t2i] = 1
            P.append(Prow)
    if len(padS)>0:             
        pad = np.vstack(padS)
        P = np.vstack(P)
    else:
        pad = np.zeros((0, 1))
        P = np.zeros((0, Bs[j].shape[0]))
    return (P, pad)

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

def direct_solve_factor(Xs, Ys, fuse_constraints, ridge_constraints, lambdaR):
    (coeff_l, con_l) = factor_constraints(Xs, Ys, fuse_constraints)
    Bs = []
        
    #Initialize matrices to hold solutions
    for i in range(len(Xs)):
        Bs.append(np.zeros((Xs[i].shape[1], Ys[i].shape[1])))
    
    for f in range(len(coeff_l)):
        #print 'subproblem %d of %d' % (f+1, len(coeff_l))
        t1 = time.time()
        coefficients = coeff_l[f]
        constraints = con_l[f]
        
        columns = set()

        for co in coefficients:
            columns.add(coefficient(co.sub, None, co.c))
        columns = list(columns)
        
        num_subproblems = len(set(map(lambda co: co.sub, coefficients)))
        
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
        #now add an appropriate number of zeros to Y_l
        Y_l.append(np.zeros((cums[-1] + len(constraints), 1)))
        
        P = np.vstack(P_l)
        X = np.vstack((diag_concat(X_l), P, I))
        y = np.vstack(Y_l)
        
        #from matplotlib import pyplot as plt
        #plt.matshow(X, aspect = float(X.shape[1])/X.shape[0])
        #plt.show()
        t2 = time.time()
        #print 'setup took %f'%(t2-t1)
        (b, resid, rank, sing) = np.linalg.lstsq(X, y)        
        
#now we put it all together - blah!
        for co_i in range(len(columns)):
            co = columns[co_i]
            start_ind = cums[co_i]
            end_ind = cums[co_i+1]
            
            Bs[co.sub][:, [co.c]] = b[start_ind:end_ind]
        t3 = time.time()
        #print 'solving took %f'%(t3-t2)
    #B2 = np.dot(np.dot(np.linalg.inv(np.dot(Xs[0].T, Xs[0])), Xs[0].T), Ys[0])
    #print B2
    return Bs
        
        

#no cleverness at all
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


def iter_rr(Xs, TFs, Ys, Gs, Cs, priorset, lamP, lamR, lamS, it):   
    Bs = []
    Is = []
    Csi = con_to_inds(TFs, Gs, Cs)
    for j in range(len(Xs)):
        Bs.append(np.zeros((Xs[j].shape[1], Ys[j].shape[1])))
        Is.append(0)#not set here
    Bsn = map(lambda x: x.copy(), Bs)
    for i in range(it):
        lamSit = lamS*float(i+1)/it        
        for j in range(len(Xs)):
            priors = priorset[j]
            for c in range(Bs[j].shape[1]):
                Is[j] = np.eye(Xs[j].shape[1])*lamR**2#here 
                for k in range(len(priors)):
                    if priors[k][1] == c:
                        Is[j][priors[k][0],priors[k][0]] = (lamP*lamR)**2     
                padR = np.zeros((Xs[j].shape[1], 1))
                (P, padS) = make_bpad(Bs, j, c, Csi)
                #now solve the regression problem
                Xc = Xs[j].copy()
                AP = np.matrix(np.vstack((Xc, P*lamSit)))
                ypad = np.matrix(np.vstack((Ys[j][:,[c]], lamSit*padS)))
                b = np.linalg.inv(AP.T * AP + Is[j])*AP.T*ypad
                Bsn[j][:, [c]] = b

        change = 0
        for j in range(len(Xs)):                 
            change += ((Bs[j] - Bsn[j])**2).sum()            

        
        tmp = Bs
        Bs = Bsn
        Bsn = tmp
    return Bs

#
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


    def factor_constraints_helper(cons, coeffs):
        coe_fr = coeffs[-1]
        #first go down the column
        for r in range(Xs[coe_fr.sub].shape[1]):
            col_nbr = coefficient(coe_fr.sub, r, coe_fr.c)
            if col_nbr in cset:
                cset.remove(col_nbr)
                coeffs.append(col_nbr)
                factor_constraints_helper(cons, coeffs)
        #now go down the constraints
        for con in cmap[coe_fr]:
            #we traverse edges in both directions, but only keep track of the ones that have the same direction we're traveling
            if con.c1 == coe_fr:
                cons.append(con)
            for coe_to in (con.c1, con.c2):
                if coe_to in cset:
                    coeffs.append(coe_to)
                    cset.remove(coe_to)
                    factor_constraints_helper(cons, coeffs)        
    coeff_l = []
    con_l = []
    while len(cset):
        coeff = cset.pop()
        coeff_l.append([coeff])
        con_l.append([])
        factor_constraints_helper(con_l[-1], coeff_l[-1])
    return (coeff_l, con_l)



def solve_from_groups(Xs, TFs, Ys, Gs, groups, priorset, lamP, lamR, lamS, it):
    Cs = make_constraints_group(groups)
    return iter_rr(Xs, TFs, Ys, Gs, Cs, priorset, lamP, lamR, lamS, it)


#makes constraints from the list of fusion groups given by groups
#groups: [[[(tf, gene),...],[(tf,gene),...]],...]
#reminder format for Cs: (i, tf1, g1, j, tf2, g2) 
#currently only works for two subproblems
def make_constraints_group(groups):
    Cs=[]
    for n in range(len(groups)):
        b1 = groups[n][0]
        b2 = groups[n][1]
        for a in range(len(b1)):
            for b in range(len(b2)):
                Cs.append((0, str(b1[a][0]), str(b1[a][1]), 1, str(b2[b][0]), str(b2[b][1])))
    return Cs
