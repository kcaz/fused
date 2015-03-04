import fused_reg as fr
import data_sources as ds
import fit_grn as fg
import numpy as np
np.random.seed(9221999)
import os
import matplotlib
#matplotlib.use('PS')
from matplotlib import pyplot as plt
import random
import pandas as pd
import seaborn as sns
sns.set(palette="Set2")
#This file is just a list of experiments. Only code that is used nowhere else is appropriate (ie general plotting code should go somewhere else

#this is basic - loads all the data, fits the basic model, and returns B
def test_bacteria(lamP, lamR, lamS):
    subt = ds.standard_source('data/bacteria_standard',0)
    anthr = ds.standard_source('data/bacteria_standard',1)

    
    (bs_e, bs_t, bs_genes, bs_tfs) = subt.load_data()
    (ba_e, ba_t, ba_genes, ba_tfs) = anthr.load_data()

    (bs_priors, bs_sign) = subt.get_priors()
    (ba_priors, ba_sign) = anthr.get_priors()


    Xs = [bs_t, ba_t]
    Ys = [bs_e, ba_e]
    genes = [bs_genes, ba_genes]
    tfs = [bs_tfs, ba_tfs]
    priors = bs_priors + ba_priors
    orth = ds.load_orth('data/bacteria_standard/orth',[anthr.name, subt.name])
    organisms = [subt.name, anthr.name]

    Bs = fr.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)

    return Bs

#load all the anthracis data and some of the subtilis data
def test_bacteria_subs_subt(lamP, lamRs, lamSs, k=20, eval_con=False):
    out = 'data/bacteria_standard'
    errds = []
    for lamR in lamRs:
        for lamS in lamSs:
            errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False, eval_con=True)
            errds.append(errd)
    return errds




#In this experiment, we generate several data sets with a moderate number of TFs and genes. Fusion is total and noiseless. Measure performance as a function of lamS, and performance as a function of the amount of data. Plot both. This is a basic sanity check for fusion helping
def sanity1():
    repeats = 1
    N_TF = 15
    N_G = 100
    data_amnts = [10, 20,30]
    lamSs = [0, 0.5,1]
    if not os.path.exists(os.path.join('data','fake_data','sanity1')):
        os.mkdir(os.path.join('data','fake_data','sanity1'))
    #iterate over how much data to use
    errors = np.zeros((len(data_amnts), len(lamSs)))
    for k in range(repeats):
        for i, N in enumerate(data_amnts):
            out = os.path.join('data','fake_data','sanity1','dat_'+str(N))
            ds.write_fake_data1(N1 = 10*N, N2 = 10*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.0)
            lamR = 0.1
            lamP = 1.0 #priors don't matter
            for j, lamS in enumerate(lamSs):
                errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
                errors[i, j] += errd['mse'][0]
    for i in range(len(data_amnts)):
        for j in range(len(lamSs)):
            errors[i,j] == errors[i,j]/10
    plt.plot(errors.T)
    plt.legend(data_amnts)
    plt.xlabel('lamS')
    plt.ylabel('mse')
    plt.savefig(os.path.join(os.path.join('data','fake_data','sanity1','fig1')))
    plt.figure()
    plt.plot(errors)
    plt.legend(lamSs)
    plt.xlabel('lamS')
    plt.ylabel('mse')
    plt.savefig(os.path.join(os.path.join('data','fake_data','sanity1','fig2')))

#shows performance as a function of the amount of  data in a secondary network for several values of lamS
def increase_data():
    repeats = 10
    N_TF = 25
    N_G = 200
    fixed_data_amt = 20
    data_amnts = [10, 20,30,40]
    lamSs = [0,1.0]#[0, 0.25,0.5,0.75,1]
    if not os.path.exists(os.path.join('data','fake_data','increasedata')):
        os.mkdir(os.path.join('data','fake_data','increasedata'))
    #iterate over how much data to use
    errors = np.zeros((len(data_amnts), len(lamSs)))
    for k in range(repeats):
        for i, N in enumerate(data_amnts):
            out = os.path.join('data','fake_data','increasedata','dat_'+str(N))
            ds.write_fake_data1(N1 = 10*fixed_data_amt, N2 = 10*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.0)
            lamR = 0.1
            lamP = 1.0 #priors don't matter
            for j, lamS in enumerate(lamSs):
                errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
                errors[i, j] += errd['mse'][0]
    for i in range(len(data_amnts)):
        for j in range(len(lamSs)):
            errors[i,j] = errors[i,j]/k
    for r, amnt in enumerate(data_amnts):
        plt.plot(data_amnts, errors[r, :])

    plt.legend(data_amnts)
    plt.savefig(os.path.join(os.path.join('data','fake_data','increasedata','fig1')))
    plt.figure()
    for c, lamS in enumerate(lamSs):
        plt.plot(lamSs, errors[:, c])
    plt.legend(lamSs)
    plt.savefig(os.path.join(os.path.join('data','fake_data','increasedata','fig2')))
    plt.show()


def test_scad():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    N_TF = 10
    N_G = 200
    amt_fused = 1.0
    orth_err = [0,0.3,0.5,1.0]
    lamSs = [0, 0.5]
    if not os.path.exists(os.path.join('data','fake_data','test_scad')):
        os.mkdir(os.path.join('data','fake_data','test_scad'))
    #iterate over how much fusion
    errors_scad = np.zeros((len(orth_err), len(lamSs)))
    errors_l2 = np.zeros((len(orth_err), len(lamSs)))
    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','test_scad','dat_'+str(N))
        ds.write_fake_data1(N1 = 10*10, N2 = 10*10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, orth_falseneg = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
        lamR = 0.1
        lamP = 1.0 #priors don't matter
        for j, lamS in enumerate(lamSs):
            errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_scad', reverse = True, special_args = {'s_it':50}, cv_both = (True, True))
            errl = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
            errors_scad[i,j] = errd['mse'][0]
            errors_l2[i,j] = errl['mse'][0]

    colorlist = [[0,0,1],[0,1,0],[1,0,0],[0.5,0,0.5]]
    for r, amnt in enumerate(orth_err):
        plt.plot(orth_err, errors_scad[r,:], color = colorlist[r])
    for r, amnt in enumerate(orth_err):
        plt.plot(orth_err, errors_l2[r,:], '--', color = colorlist[r])
    #plt.legend(lamSs+lamSs)
    plt.xlabel('orth error')
    plt.ylabel('mean squared error')
    plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad','fig3')))
    plt.figure()


def test_scad2():
#create simulated data set with false orthology and run fused scad + visualize scad penalty at each cv
     
    N_TF = 10
    N_G = 200
    amt_fused = 1.0
    orth_err = 0.3
    lamS = 0.5

    if not os.path.exists(os.path.join('data','fake_data','test_scad2')):
        os.mkdir(os.path.join('data','fake_data','test_scad2'))

    out = os.path.join('data','fake_data','test_scad2','ortherr'+str(orth_err))
    ds.write_fake_data1(N1 = 10*10, N2 = 10*10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = orth_err, orth_falseneg = orth_err, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
    lamR = 0.1
    lamP = 1.0 #priors don't matter

    k = 10
    ds1 = ds.standard_source(out,0)
    ds2 = ds.standard_source(out,1)
    organisms = [ds1.name, ds2.name]
    orth_fn = os.path.join(out, 'orth')
    orth = ds.load_orth(orth_fn, organisms)

    folds1 = ds1.partition_data(10)
    folds2 = ds2.partition_data(10)
    excl = lambda x,i: x[0:i]+x[(i+1):]
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()

    (e1, t1, genes1, tfs1) = ds1.load_data()
    (e2, t2, genes2, tfs2) = ds2.load_data()
    fuse_constraints = fr.orth_to_constraints(organisms, [genes1, genes2], [tfs1, tfs2], orth, lamS)

    k = 1
    for fold in range(k):
        deltabeta = []
        fusionpenalty = []
        lamS = []

        #get conditions for current cross-validation fold
        f1_te_c = folds1[fold]
        f1_tr_c = np.hstack(excl(folds1, fold))

        f2_te_c = folds2[fold]
        f2_tr_c = np.hstack(excl(folds2, fold))

        (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data(f1_tr_c)
        (e1_te, t1_te, genes1, tfs1) = ds1.load_data(f1_te_c)

        (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data(f2_tr_c)
        (e2_te, t2_te, genes2, tfs2) = ds2.load_data(f2_te_c)

        Xs = [t1_tr, t2_tr]
        Ys = [e1_tr, e2_tr]
        genes = [genes1, genes2]
        tfs = [tfs1, tfs2]
        priors = priors1 + priors2

        Bs_unfused = fr.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP=1.0, lamR=0.1, lamS=0)
        fuse_constraints = fr.orth_to_constraints(organisms, genes, tfs, orth, lamS)
        print fuse_constraints
        new_fuse_constraints = fr.scad(Bs_unfused, fuse_constraints, lamS, lamW=None, a=0.5)

        #get delta beta for each fusion constraint pair
        for pair in new_fuse_constraints:
            #print pair
            r1 = pair.c1.r
            c1 = pair.c1.c
            r2 = pair.c2.r
            c2 = pair.c2.c
            B1u = Bs_unfused[0][r1][c1]
            B2u = Bs_unfused[1][r2][c2]
            deltabeta.append(B1u-B2u)
            s = pair.lam
            #print s
            fusionpenalty.append(s*((B1u-B2u)**2))
            lamS.append(s)

#        plt.scatter(deltabeta, fusionpenalty)
#        plt.xlabel('delta beta')
#        plt.ylabel('penalty')
#        plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad2',str(fold))))
#        plt.figure()
#        plt.clf()

#        plt.hist(lamS)
#        plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad2',str(fold)+'lamS')))



def test_mcp():
#create simulated data set with false orthology and run fused scad + visualize mcp penalty at each cv
     
    N_TF = 10
    N_G = 200
    amt_fused = 1.0
    orth_err = 0.3
    lamS = 0.25

    if not os.path.exists(os.path.join('data','fake_data','test_mcp')):
        os.mkdir(os.path.join('data','fake_data','test_mcp'))

    out = os.path.join('data','fake_data','test_mcp','ortherr'+str(orth_err))
    ds.write_fake_data1(N1 = 10*10, N2 = 10*10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = orth_err, orth_falseneg = orth_err, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
    lamR = 0.1
    lamP = 1.0 #priors don't matter

    k = 10
    ds1 = ds.standard_source(out,0)
    ds2 = ds.standard_source(out,1)
    organisms = [ds1.name, ds2.name]
    orth_fn = os.path.join(out, 'orth')
    orth = ds.load_orth(orth_fn, organisms)

    folds1 = ds1.partition_data(10)
    folds2 = ds2.partition_data(10)
    excl = lambda x,i: x[0:i]+x[(i+1):]
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()

    (e1, t1, genes1, tfs1) = ds1.load_data()
    (e2, t2, genes2, tfs2) = ds2.load_data()
    fuse_constraints = fr.orth_to_constraints(organisms, [genes1, genes2], [tfs1, tfs2], orth, lamS)

    for fold in range(k):
        deltabeta = []
        fusionpenalty = []

        #get conditions for current cross-validation fold
        f1_te_c = folds1[fold]
        f1_tr_c = np.hstack(excl(folds1, fold))

        f2_te_c = folds2[fold]
        f2_tr_c = np.hstack(excl(folds2, fold))

        (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data(f1_tr_c)
        (e1_te, t1_te, genes1, tfs1) = ds1.load_data(f1_te_c)

        (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data(f2_tr_c)
        (e2_te, t2_te, genes2, tfs2) = ds2.load_data(f2_te_c)

        Xs = [t1_tr, t2_tr]
        Ys = [e1_tr, e2_tr]
        genes = [genes1, genes2]
        tfs = [tfs1, tfs2]
        priors = priors1 + priors2

        Bs_unfused = fr.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP=1.0, lamR=0.1, lamS=0)
        new_fuse_constraints = fr.mcp(Bs_unfused, fuse_constraints, lamS, lamW=None, a=0.5)

        #get delta beta for each fusion constraint pair
        for pair in new_fuse_constraints:
            r1 = pair.c1.r
            c1 = pair.c1.c
            r2 = pair.c2.r
            c2 = pair.c2.c
            B1u = Bs_unfused[0][r1][c1]
            B2u = Bs_unfused[1][r2][c2]
            deltabeta.append(B1u-B2u)
            
        for pair in new_fuse_constraints:
            r1 = pair.c1.r
            c1 = pair.c1.c
            s = pair.lam
            r2 = pair.c2.r
            c2 = pair.c2.c
            B1f = Bs_unfused[0][r1][c1]
            B2f = Bs_unfused[1][r2][c2]
            fusionpenalty.append(s*(B1f-B2f)**2)

        plt.scatter(deltabeta, fusionpenalty)
        plt.xlabel('delta beta')
        plt.ylabel('penalty')
        plt.savefig(os.path.join(os.path.join('data','fake_data','test_mcp',str(fold))))
        plt.figure()


def studentseminar():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    lamSs = [0, 0.25,0.5,0.75,1]
    errors = np.zeros(len(lamSs))
    for i, N in enumerate(lamSs):
        out = os.path.join('data','bacteria_standard')
        lamR = 0.1
        lamP = 1.0 #priors don't matter
        errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=N, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, False))
        errors[i] = errd['aupr'][0]
    
    plt.plot(lamSs, errors, 'ro')
    plt.savefig(os.path.join(os.path.join('data','bacteria_standard','studentseminar','fig1')))
    plt.figure()


#generates two fused beta vectors, with a fusion constraint on each pair.
#plots constraints after scad adjustment
def test_scad1(lamS=1, a=3.7):
    b1 = np.linspace(0,5,100)[:, None] 
    b2 = np.zeros(b1.shape)

    #(Bs_init, fuse_constraints, lamS, a):
    
    constraints = []
    for i in range(b1.shape[0]):
        con = fr.constraint(fr.coefficient(0,i,0), fr.coefficient(1,i,0), lamS)
        constraints.append(con)
    new_cons = fr.scad([b1, b2], constraints, lamS, a)
    new_con_val = np.array(map(lambda x: x.lam, new_cons))
    plt.plot(b1, new_con_val)
    plt.show()



#tests fusion visually using a pair of similar two-coefficient networks
def test_2coeff_fuse():
    lamPs = np.array([1])
    lamRs = np.array([0.1])
    lamSs = np.linspace(0,2,10)
    
    out = os.path.join('data','fake_data','2coeff_fuse')
    if not os.path.exists(out):
        os.mkdir(out)

    N = 10
    ds.write_fake_data1(out_dir=out, N1=N,N2=N,tfg_count1=(2,1),tfg_count2=(2,1),sparse=0.0,fuse_std=0.0,measure_noise1=0.3,measure_noise2=0.3)
    

    (br1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (br2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
    
    for lamS in lamSs:
    
    #fg.cv_model1(data_fn = out, lamP=lamPs[0], lamR=lamRs[0], lamS=lamSs[i], k= 10)
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS)
        plt.plot(b1[0,2], b1[1,2], 'or',markersize=5+10*lamS)
        plt.plot(b2[0,2], b2[1,2], 'ob',markersize=5+10*lamS)
        #print b1
        #print br1

    plt.plot(br1[0,2], br1[1,2], '*',c=[0.5,0,0.5],markersize=30)
    
    plt.plot(br2[0,2], br2[1,2], '*',c=[0.5,0,0.5],markersize=30)
    
    plt.rcParams.update({'font.size': 18})
    plt.axis('equal')
    plt.xlabel('coefficient1')
    plt.ylabel('coefficient2')
    plt.show()


#tests fusion visually using a pair of similar two-coefficient networks
#this network only has 50% ortho coverage
def test_2coeff_fuse_H():
    lamPs = np.array([1])
    lamRs = np.array([0.1])
    lamSs = np.linspace(0,2,10)
    
    out = os.path.join('data','fake_data','2coeff_fuse_H')
    if not os.path.exists(out):
        os.mkdir(out)

    N = 10
    ds.write_fake_data1(out_dir=out, N1=N,N2=N,tfg_count1=(2,1),tfg_count2=(2,1),sparse=0.0,fuse_std=0.1,measure_noise1=0.3,measure_noise2=0.3, pct_fused=0.5, orth_falsepos=0.99)#think there are orths that are not
    

    (br1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (br2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
    plt.plot(br1[0,2], br1[1,2], '*',c='r',markersize=30)
    
    plt.plot(br2[0,2], br2[1,2], '*',c='b',markersize=30)
    
    for lamS in lamSs:
    
        #fg.cv_model1(data_fn = out, lamP=lamPs[0], lamR=lamRs[0], lamS=lamSs[i], k= 10)
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS)
        plt.plot(b1[0,2], b1[1,2], 'or',markersize=5+10*lamS)
        plt.plot(b2[0,2], b2[1,2], 'ob',markersize=5+10*lamS)
        #print b1
        #print br1

    
    plt.rcParams.update({'font.size': 18})
    plt.axis('equal')
    plt.xlabel('coefficient1')
    plt.ylabel('coefficient2')
    plt.show()
    
#tests fusion visually using a pair of similar two-coefficient networks
#one network has a lot more data
def test_2coeff_fuse1B():
    lamPs = np.array([1])
    lamRs = np.array([0.1])
    lamSs = np.linspace(0,2,10)
    
    out = os.path.join('data','fake_data','2coeff_fuse1B')
    if not os.path.exists(out):
        os.mkdir(out)

    N1 = 10
    N2 = 1000
    ds.write_fake_data1(out_dir=out, N1=N1,N2=N2,tfg_count1=(2,1),tfg_count2=(2,1),sparse=0.0,fuse_std=0.0,measure_noise1=0.3,measure_noise2=0.3)
    

    (br1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (br2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
    plt.plot(br1[0,2], br1[1,2], '*',c=[0.5,0,0.5],markersize=30)
    
    plt.plot(br2[0,2], br2[1,2], '*',c=[0.5,0,0.5],markersize=30)
    
    for lamS in lamSs:
    
    #fg.cv_model1(data_fn = out, lamP=lamPs[0], lamR=lamRs[0], lamS=lamSs[i], k= 10)
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS)
        plt.plot(b1[0,2], b1[1,2], 'or',markersize=5+10*lamS)
        plt.plot(b2[0,2], b2[1,2], 'ob',markersize=5+10*lamS)
        #print b1
        #print br1

    
    plt.rcParams.update({'font.size': 18})
    plt.axis('equal')
    plt.xlabel('coefficient1')
    plt.ylabel('coefficient2')
    plt.show()


#tests fusion visually using a pair of similar two-coefficient networks
#this network only has 50% ortho coverage
#NOW USES SCAD!
def test_2coeff_fuse_HS():
    lamPs = np.array([1])
    lamRs = np.array([0.1])
    lamSs = 1.0-np.linspace(0,1,10)
    
    out = os.path.join('data','fake_data','2coeff_fuse_HS')
    if not os.path.exists(out):
        os.mkdir(out)

    N = 5
    ds.write_fake_data1(out_dir=out, N1=N,N2=N,tfg_count1=(2,1),tfg_count2=(2,1),sparse=0.0,fuse_std=0.0,measure_noise1=0.1,measure_noise2=0.1, pct_fused=0.5, orth_falsepos=0.99)#think there are orths that are not    

    (br1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (br2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
    plt.plot(br1[0,2], br1[1,2], '*',c=[1.0,0,0],markersize=30)
    
    plt.plot(br2[0,2], br2[1,2], '*',c=[0,0,1.0],markersize=30)
    #we want to grab the fusion constraints that affect the coefficients we care about, specifically 0->2 <--> 0'->2', 1->2 <--> 1' -> 2'
    def assemble_orths(orths):
        fusemat = np.zeros((2,1))
        
        for orth in orths:
            if orth.c1.sub == 1 and orth.c1.c == 2:
                fusemat[orth.c1.r, 0] = orth.lam
                #print '%d, %d fused to %d, %d - lamS = %f' %(orth.c1.r, orth.c1.c, orth.c2.r, orth.c2.c, orth.lam)
        print fusemat
    for lamS in lamSs:
        print 'SOLVING'
        #fg.cv_model1(data_fn = out, lamP=lamPs[0], lamR=lamRs[0], lamS=lamSs[i], k= 10)
        special_args = {'s_it':5, 'orths':None, 'a':0.1}
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS, solver='solve_ortho_direct_scad',special_args = special_args)
        plt.plot(b1[0,2], b1[1,2], 'or',markersize=0.5*(10+20*lamS))
        plt.plot(b2[0,2], b2[1,2], 'ob',markersize=0.5*(10+20*lamS))
        
        assemble_orths(special_args['orths'])
        #print b1
        #print br1
        
    (constraints, marks, orths) = ds.load_constraints(out)
    for i, orth in enumerate(orths):
        print orth
    for i, con in enumerate(constraints):
        print '%s    %s' % (str(con), str(marks[i]))
    

    plt.rcParams.update({'font.size': 18})
    plt.axis('equal')
    plt.xlabel('coefficient1')
    plt.ylabel('coefficient2')
    plt.savefig(os.path.join(out,'fig1'))
    plt.show()


#tests fusion visually using a pair of similar two-coefficient networks
#this network only has 50% ortho coverage
#NOW USES SCAD!
def test_2coeff_fuse_HS2():
    lamPs = np.array([1])
    lamRs = np.array([0.1])
    lamSs = 1.0-np.linspace(0,1,10)
    n_tfs = 4
    out = os.path.join('data','fake_data','2coeff_fuse_HS2')
    if not os.path.exists(out):
        os.mkdir(out)
    
    N = 30
    ds.write_fake_data1(out_dir=out, N1=N,N2=N,tfg_count1=(n_tfs,1),tfg_count2=(n_tfs,1),sparse=0.0,fuse_std=0.0,measure_noise1=0.0,measure_noise2=0.0, pct_fused=0.5, orth_falsepos=0.99)#think there are orths that are not
    

    (br1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (br2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
    (constraints, marks, orths) = ds.load_constraints(out)
    
    #now we want a True gene targeting constraint to be axis1, and a false gene targeting constraint to be axis2
    inds = np.arange(len(marks))


    con_true = set(inds[np.array(marks) == True])
    con_false = set(inds[np.array(marks) == False])


    con_gene_targ1 = set(inds[np.array(map(lambda x: x.c1.c >= br1.shape[0] and x.c1.sub == 0, constraints))]) #genes come after tfs
    con_gene_targ2 = set(inds[np.array(map(lambda x: x.c1.c >= br2.shape[0] and x.c1.sub == 1, constraints))]) #genes come after tfs
    con_gene_target = con_gene_targ1.union(con_gene_targ2)
    

    true_gene_targets = list(con_true.intersection(con_gene_target))
    false_gene_targets = list(con_false.intersection(con_gene_target))
    
    axis1con = constraints[random.choice(true_gene_targets)]
    axis2con = constraints[random.choice(false_gene_targets)]

    #for b1
    ax1r1 = axis1con.c1.r
    ax1c1 = axis1con.c1.c
    ax2r1 = axis2con.c1.r
    ax2c1 = axis2con.c1.c

    #for b2
    ax1r2 = axis1con.c2.r
    ax1c2 = axis1con.c2.c
    ax2r2 = axis2con.c2.r
    ax2c2 = axis2con.c2.c
    
#we want to grab the fusion constraints that affect the coefficients we care about, specifically 0->2 <--> 0'->2', 1->2 <--> 1' -> 2'




    if False:
        for i, orth in enumerate(orths):
            print orth
        for i, con in enumerate(constraints):
            print '%s    %s' % (str(con), str(marks[i]))

    plt.close()
    plt.plot(br1[ax1r1,ax1c1], br1[ax2r1,ax2c1], '*',c=[1.0,0,0],markersize=30)
    
    plt.plot(br2[ax1r2,ax1c2], br2[ax2r2,ax2c2], '*',c=[0,0,1.0],markersize=30)
    for lamS in lamSs:
        print 'SOLVING'
        #fg.cv_model1(data_fn = out, lamP=lamPs[0], lamR=lamRs[0], lamS=lamSs[i], k= 10)
        special_args = {'s_it':5, 'orths':None, 'a':1.50}
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS, solver='solve_ortho_direct_scad',special_args = special_args)
        plt.plot(b1[ax1r1,ax1c1], b1[ax2r1,ax2c1], 'or',markersize=0.5*(10+20*lamS))
        plt.plot(b2[ax1r2,ax1c2], b2[ax2r2,ax2c2], 'ob',markersize=0.5*(10+20*lamS))
    
        
    
    plt.rcParams.update({'font.size': 18})
    plt.axis([-3, 3, -3, 3])
    #plt.axis('equal')
    plt.xlabel('coefficient1')
    plt.ylabel('coefficient2')
    #plt.savefig(os.path.join(out,'fig1'))
    plt.show()

#constructs small, 100% fused, models and profiles the solver
def bench1():
    N_TF = 50
    N_G = 100
    N1 = 40
    N2 = 40
    out = os.path.join('data','fake_data','bench1')
    #ds.write_fake_data1(N1 = N1, N2 = N2, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.0)
    def waka():        
        (b1, b2) = fg.fit_model(out, 1.0, 0.1, 0.5, solver='solve_ortho_direct')
    import profile
    profile.runctx("waka()",globals(),locals(),sort='cumulative',filename=os.path.join(out,'stats'))
    import pstats
    p = pstats.Stats(os.path.join(out,'stats'))
    p.sort_stats('cumulative').print_stats(20)

#solves a small problem and compares to reference (slow) solver
def ref1():
    N_TF = 10
    N_G = 2
    N1 = 6
    N2 = 6
    out = os.path.join('data','fake_data','ref1')
    ds.write_fake_data1(N1 = N1, N2 = N2, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.0,pct_fused=1.0)
    
    (b1, b2) = fg.fit_model(out, 1.0, 0.1, 0.5, solver='solve_ortho_direct')
    (n11,g,t) = ds.load_network(os.path.join(out,'beta1'))
    (n12,g,t) = ds.load_network(os.path.join(out,'beta2'))
    
    (b1r, b2r) = fg.fit_model(out, 1.0, 0.1, 0.5, solver='solve_ortho_ref')
    (n21,g,t) = ds.load_network(os.path.join(out,'beta1'))
    (n22,g,t) = ds.load_network(os.path.join(out,'beta2'))
    #from matplotlib import pyplot as plt
    #plt.matshow(n11)
    #plt.matshow(n22)
    print 'error1 %f' % ((n11-n21)**2).sum()
    print 'error2 %f' % ((n12-n22)**2).sum()

#solves a simple model and computes aupr, then does it again with priors turned on
#NOTE: it seems like adding priors always hurts network recovery. why might this be? 
def check_structure1():
    N_TF = 20
    N_G = 50
    N1 = 20
    N2 = 20
    sparse = 0.5
    out = os.path.join('data','fake_data','struct1')
    ds.write_fake_data1(N1 = N1, N2 = N2, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=sparse, fuse_std = 0.0)
    
    lamP = 1.0
    lamR = 100
    lamS = 0.0001
    k=5
    solver='solve_ortho_direct'
    reverse=False
    cv_both = (True, True)
    errd1 = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
    lamP = 0.01
    errd2 = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
    print errd1
    print errd2
    (b1, b2) = fg.fit_model(out, lamP, lamR, lamS, solver='solve_ortho_direct')
    (b1r, g, t) = ds.load_network(os.path.join(out, 'beta1'))
    return (b1, b2)

#solves a simple model and computes aupr, then does it again with fusion turned on
#also looks at the beta error
def check_structure2():
    N_TF = 20
    N_G = 50
    N1 = 10
    N2 = 10
    sparse = 0.5
    out = os.path.join('data','fake_data','struct2')
    ds.write_fake_data1(N1 = N1, N2 = N2, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=sparse, fuse_std = 0.0)
    
    lamP = 1.0
    lamR = 1
    lamS = 0.0
    k=5
    solver='solve_ortho_direct'
    reverse=False
    cv_both = (True, True)
    errd1 = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
    lamS = 1
    
    errd2 = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
    print errd1
    print errd2
    (b1, b2) = fg.fit_model(out, lamP, lamR, lamS=0, solver='solve_ortho_direct')
    (b1S, b2S) = fg.fit_model(out, lamP, lamR, lamS=lamS, solver='solve_ortho_direct')
    (b1r, g, t) = ds.load_network(os.path.join(out, 'beta1'))
    print 'beta error 1 %f'% fg.eval_network_beta(b1, b1r)
    print 'beta error 2 %f'% fg.eval_network_beta(b1S, b1r)
#return (b1, b2)

#generates simple model, then inspects priors
def make_sure_priors_are_right():
    N_TF = 3
    N_G = 3
    N1 = 15
    N2 = 15
    sparse = 0.5
    out = os.path.join('data','fake_data','check_priors')
    ds.write_fake_data1(N1 = N1, N2 = N2, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=sparse, fuse_std = 0.0)
    
    d1 = ds.standard_source(out,0)
    d2 = ds.standard_source(out,1)
    
    (p1,s1) = d1.get_priors()
    (p2,s2) = d2.get_priors()
    
    p1c = fr.priors_to_constraints([d1.name],[d1.genes],[d1.tfs],p1,0.5)
    p2c = fr.priors_to_constraints([d2.name],[d2.genes],[d2.tfs],p2,0.5)

    (b1r, g, t) = ds.load_network(os.path.join(out, 'beta1'))
    (b2r, g, t) = ds.load_network(os.path.join(out, 'beta2'))


    print '\n'.join(map(str, p1))

    print '\n'.join(map(str, p1c))
    print b1r
    print '\n'.join(map(str, p2c))
    print b2r
    r_roc = fg.eval_network_roc(b1r, d1.genes, d1.tfs, p1, exclude_tfs=True)
    print 'roc is %f' % r_roc

#look at unfused performance varying with lamR
def vary_lamR():
    repeats = 5
    N_TF = 25
    N_G = 50
    
    N = 100
    lamRs = np.linspace(0.00001,10,7)
    lamSs = [0,0.5]
    out1 = os.path.join('data','fake_data','vary_lamR')
    k = 5#cv folds
    if not os.path.exists(out1):
        os.mkdir(out1)
    #iterate over how much data to use
    errors = np.zeros((2, len(lamRs)))
    for r in range(repeats):
        
        out2 = os.path.join(out1,'dat_'+str(N))
        ds.write_fake_data1(N1 = k*N, N2 = k*N, out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.0)
            
        lamP = 1.0 #priors don't matter
        for j, lamR in enumerate(lamRs):
            for i, lamS in enumerate(lamSs):
                errd = fg.cv_model1(out2, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
                errors[i, j] += errd['mse'][0]
    
    errors = errors / k
    
    plt.close()
    
    plt.plot(lamRs, errors[0,:])
    plt.plot(lamRs, errors[1,:])
    
    plt.savefig(os.path.join(out2, 'fig'))
    plt.show()


#we want to show performance as a function of data amount for lamS=0, lamS=1
#we are going to try and set lamR to its optimum
# one network has a fixed number of conditions (20) the other network varies
def increase_data2():
    repeats = 5
    N_TF = 25
    N_G = 50
    N_fixed = 10
    N_varies = np.arange(10,41,2)#[10,20,30,40]


    lamR = 0.1
    lamSs = [0,0.5]
    out1 = os.path.join('data','fake_data','increase_data2')
    k = 5#cv folds
    if not os.path.exists(out1):
        os.mkdir(out1)
    #iterate over how much data to use
    errors = np.zeros((2, len(N_varies)))
    for r in range(repeats):
        for j, N in enumerate(N_varies):
            out2 = os.path.join(out1,'dat_'+str(N))
            ds.write_fake_data1(N1 = k*N, N2 = k*N_fixed, out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1, pct_fused=0.75)
            
            lamP = 1.0 #priors don't matter
            for i, lamS in enumerate(lamSs):
                errd = fg.cv_model1(out2, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
                errors[i, j] += errd['R2'][0]
    
    errors = errors / k
    
    plt.close()
    
    plt.plot(N_varies, errors[0,:])
    plt.plot(N_varies, errors[1,:])
    plt.legend(lamSs)
    plt.savefig(os.path.join(out2, 'fig'))

    plt.rcParams.update({'font.size': 18})

    #plt.axis('equal')
    plt.xlabel('conditions')
    plt.ylabel('R2')
    plt.savefig(os.path.join(out1,'fig'))

    plt.show()

#lets us look at the delta betas for random betas and for betas with fusion constraints, plots them before and after using em and identifies 
def plot_betas():
    N_TF = 50
    N_G = 1000
    lamP = 1.0
    lamR = 1.0
    lamS = 1.0

    out1 = os.path.join('data','fake_data','test_em1')
    if not os.path.exists(out1):
        os.mkdir(out1)
    out2 = os.path.join(out1,'dat')
    ds.write_fake_data1(N1 = 40, N2 = 40, out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, orth_falsepos=0.3, orth_falseneg=0.3, fuse_std = 0.1, pct_fused=0.75)
    ds1 = ds.standard_source(out2,0)
    ds2 = ds.standard_source(out2,1)
    orth_fn = os.path.join(out2, 'orth')
    organisms = [ds1.name, ds2.name]
    orth = ds.load_orth(orth_fn, organisms)
    (e1, t1, genes1, tfs1) = ds1.load_data()
    (e2, t2, genes2, tfs2) = ds2.load_data()
    Xs = [t1, t2]
    Ys = [e1, e2]
    genes = [genes1, genes2]
    tfs = [tfs1, tfs2]
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()	
    priors = priors1 + priors2
    (constraints, marks) = fr.orth_to_constraints_marked(organisms, genes, tfs, orth, 1.0)

    Bs_uf = fr.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS=0)
    Bs_fr = fr.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)
    s_it = 5
    Bs_fs = fr.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, s_it)
    em_it=5    
    special_args={'f':1,'uf':1}
    Bs_em = fr.solve_ortho_direct_em(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, em_it, special_args=special_args)
    r1 = Bs_uf[0].shape[0]
    c1 = Bs_uf[0].shape[1]
    r2 = Bs_uf[1].shape[0]
    c2 = Bs_uf[1].shape[1]
    Buf1 = []
    Buf2 = []
    Bfr1 = []
    Bfr2 = []
    Bfs1 = []
    Bfs2 = []
    Bem1 = []
    Bem2 = []
    colors = []
    con_inds = np.random.permutation(range(len(constraints)))
    area_fs = []
    area_em = [] 

    subs = 300
    for i in con_inds[0:subs]:
        con = constraints[i]
        mark = marks[i]
        Buf1.append(Bs_uf[con.c1.sub][con.c1.r, con.c1.c])
        Buf2.append(Bs_uf[con.c2.sub][con.c2.r, con.c2.c])
        Bfr1.append(Bs_fr[con.c1.sub][con.c1.r, con.c1.c])
        Bfr2.append(Bs_fr[con.c2.sub][con.c2.r, con.c2.c])
        Bfs1.append(Bs_fs[con.c1.sub][con.c1.r, con.c1.c])
        Bfs2.append(Bs_fs[con.c2.sub][con.c2.r, con.c2.c])
        Bem1.append(Bs_em[con.c1.sub][con.c1.r, con.c1.c])
        Bem2.append(Bs_em[con.c2.sub][con.c2.r, con.c2.c])

        area_fs.append(con.lam)
        area_em.append(con.lam)
        if mark == 1:
            colors.append('g')
        else:
            colors.append('r')
        Buf1.append(Bs_uf[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)])
        Buf2.append(Bs_uf[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bfr1.append(Bs_fr[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)])
        Bfr2.append(Bs_fr[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bfs1.append(Bs_fs[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)])
        Bfs2.append(Bs_fs[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bem1.append(Bs_em[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)])
        Bem2.append(Bs_em[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        colors.append('b')
        area_fs.append(1)
        area_em.append(1)

    Buf1s = np.array(Buf1)
    Buf2s = np.array(Buf2)
    Bfr1s = np.array(Bfr1)
    Bfr2s = np.array(Bfr2)
    Bfs1s = np.array(Bfs1)
    Bfs2s = np.array(Bfs2)
    Bem1s = np.array(Bem1)
    Bem2s = np.array(Bem2)
    colors = np.array(colors)
    plt.scatter(Buf1s, Buf2s, c=colors, alpha=0.5)
    plt.savefig(os.path.join(out1,'unfused'))
    plt.scatter(Bfr1s, Bfr2s, c=colors, alpha=0.5)
    plt.savefig(os.path.join(out1,'fused l2'))
    plt.scatter(Bfs1s, Bfs2s, c=colors, alpha=0.5)
    plt.savefig(os.path.join(out1,'scad'))
    plt.scatter(Bem1s, Bem2s, c=colors, alpha=0.5)
    plt.savefig(os.path.join(out1,'em'))

    plt.show()


#lets us look at the delta betas for random betas and for betas with fusion constraints, plots them before and after using em and identifies 
#uses seaborn
def plot_betas2():
    N_TF = 20
    N_G = 500
    lamP = 1.0
    lamR = 1.0
    lamS = 1.0
    lamSe = 0.1

    out1 = os.path.join('data','fake_data','plot_betas2')
    if not os.path.exists(out1):
        os.mkdir(out1)
    out2 = os.path.join(out1,'dat')
    ds.write_fake_data1(N1 = 15, N2 = 15, out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, orth_falsepos=0.3, orth_falseneg=0.3, fuse_std = 0.1, pct_fused=0.75)
    ds1 = ds.standard_source(out2,0)
    ds2 = ds.standard_source(out2,1)
    orth_fn = os.path.join(out2, 'orth')
    organisms = [ds1.name, ds2.name]
    orth = ds.load_orth(orth_fn, organisms)
    (e1, t1, genes1, tfs1) = ds1.load_data()
    (e2, t2, genes2, tfs2) = ds2.load_data()
    Xs = [t1, t2]
    Ys = [e1, e2]
    genes = [genes1, genes2]
    tfs = [tfs1, tfs2]
    (priors1, signs1) = ds1.get_priors()
    (priors2, signs2) = ds2.get_priors()	
    priors = priors1 + priors2
    (constraints, marks) = fr.orth_to_constraints_marked(organisms, genes, tfs, orth, 1.0)

    Bs_uf = fr.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS=0)
    Bs_fr = fr.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)
    s_it = 5
    special_args_fs={'a':0.2, 'orths':None}
    Bs_fs = fr.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamSe, s_it, special_args=special_args_fs)
    m_it = 5
    special_args_mcp={'orths':None}
    Bs_fm = fr.solve_ortho_direct_mcp(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamSe, m_it, special_args=special_args_mcp)
    em_it=10
    special_args_em={'f':1,'uf':0.1}
    Bs_em = fr.solve_ortho_direct_em(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, em_it, special_args=special_args_em)
    r1 = Bs_uf[0].shape[0]
    c1 = Bs_uf[0].shape[1]
    r2 = Bs_uf[1].shape[0]
    c2 = Bs_uf[1].shape[1]
    Buf1 = []
    Buf2 = []
    Bfr1 = []
    Bfr2 = []
    Bfs1 = []
    Bfs2 = []
    Bfm1 = []
    Bfm2 = []
    Bem1 = []
    Bem2 = []

    Bufd = []
    Bfrd = []
    Bfsd = []
    Bfmd = []
    Bemd = []

    Bufu = []
    Bfru = []
    Bfsu = []
    Bfmu = []
    Bemu = []

    colors_ufem = []
    colors_fs = []
    colors_fm = []
    con_inds = np.random.permutation(range(len(constraints)))
    area_fr = []
    area_fs = []
    area_fm = []

    subs = 1000
    for i in con_inds[0:subs]:
        con = constraints[i]
        mark = marks[i]
        Buf1.append(Bs_uf[con.c1.sub][con.c1.r, con.c1.c])
        Buf2.append(Bs_uf[con.c2.sub][con.c2.r, con.c2.c])
        Bufd.append(Bs_uf[con.c1.sub][con.c1.r, con.c1.c]-Bs_uf[con.c2.sub][con.c2.r, con.c2.c])
        Bfr1.append(Bs_fr[con.c1.sub][con.c1.r, con.c1.c])
        Bfr2.append(Bs_fr[con.c2.sub][con.c2.r, con.c2.c])
        Bfrd.append(Bs_fr[con.c1.sub][con.c1.r, con.c1.c]-Bs_fr[con.c2.sub][con.c2.r, con.c2.c])

        Bem1.append(Bs_em[con.c1.sub][con.c1.r, con.c1.c])
        Bem2.append(Bs_em[con.c2.sub][con.c2.r, con.c2.c])
        Bemd.append(Bs_em[con.c1.sub][con.c1.r, con.c1.c]-Bs_em[con.c2.sub][con.c2.r, con.c2.c])
    
        if mark == 1:
            colors_ufem.append('g')

        else:
            colors_ufem.append('r')

    for i in con_inds[0:subs]:
        con = special_args_fs['orths'][i]
        mark = marks[i]
        Bfs1.append(Bs_fs[con.c1.sub][con.c1.r, con.c1.c])
        Bfs2.append(Bs_fs[con.c2.sub][con.c2.r, con.c2.c])
        Bfsd.append(Bs_fs[con.c1.sub][con.c1.r, con.c1.c]-Bs_fs[con.c2.sub][con.c2.r, con.c2.c])
        if mark == 1:
            colors_fs.append('g')
        else:
            colors_fs.append('r')
        area_fs.append(con.lam*100)

    for i in con_inds[0:subs]:
        con = special_args_fs['orths'][i]
        mark = marks[i]
        Bfm1.append(Bs_fm[con.c1.sub][con.c1.r, con.c1.c])
        Bfm2.append(Bs_fm[con.c2.sub][con.c2.r, con.c2.c])
        Bfmd.append(Bs_fm[con.c1.sub][con.c1.r, con.c1.c]-Bs_fm[con.c2.sub][con.c2.r, con.c2.c])
        area_fr.append(con.lam)
        area_fs.append(con.lam)
        area_fm.append(con.lam)

        if mark == 1:
            colors_fm.append('g')
        else:
            colors_fm.append('r')
        area_fm.append(con.lam*100)

        Buf1.append(Bs_uf[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)])
        Buf2.append(Bs_uf[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bufu.append(Bs_uf[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)]-Bs_uf[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bfr1.append(Bs_fr[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)])
        Bfr2.append(Bs_fr[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bfru.append(Bs_fr[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)]-Bs_fr[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bfs1.append(Bs_fs[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)])
        Bfs2.append(Bs_fs[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bfsu.append(Bs_fs[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)]-Bs_fs[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bfm1.append(Bs_fm[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)])
        Bfm2.append(Bs_fm[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bfmu.append(Bs_fm[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)]-Bs_fm[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bem1.append(Bs_em[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)])
        Bem2.append(Bs_em[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        Bemu.append(Bs_em[0][random.randrange(0,r1,1)][random.randrange(0,c1,1)]-Bs_em[1][random.randrange(0,r2,1)][random.randrange(0,c2,1)])
        colors_ufem.append('b')
        colors_fs.append('b')
        colors_fm.append('b')

        area_fr.append(10)
        area_fs.append(10)
        area_fm.append(10)

    Buf1s = np.array(Buf1)
    Buf2s = np.array(Buf2)
    Bfr1s = np.array(Bfr1)
    Bfr2s = np.array(Bfr2)
    Bfs1s = np.array(Bfs1)
    Bfs2s = np.array(Bfs2)
    Bfm1s = np.array(Bfm1)
    Bfm2s = np.array(Bfm2)
    Bem1s = np.array(Bem1)
    Bem2s = np.array(Bem2)
    colors_ufem = np.array(colors_ufem)
    colors_fs = np.array(colors_fs)
    colors_fm = np.array(colors_fm)

    plt.close()
    sns.kdeplot(np.array(Bufd), shade=True)
    sns.kdeplot(np.array(Bufu), shade=True)
    plt.savefig(os.path.join(out1,'unfuseddist'))
    plt.close()
    sns.kdeplot(np.array(Bfrd), shade=True)
    sns.kdeplot(np.array(Bfru), shade=True)
    plt.savefig(os.path.join(out1,'ridgedist'))
    plt.close()
    sns.kdeplot(np.array(Bfsd), shade=True)
    sns.kdeplot(np.array(Bfsu), shade=True)
    plt.savefig(os.path.join(out1,'scaddist'))
    plt.close()
    sns.kdeplot(np.array(Bfmd), shade=True)
    sns.kdeplot(np.array(Bfmu), shade=True)
    plt.savefig(os.path.join(out1,'mcpdist'))
    plt.close()
    sns.kdeplot(np.array(Bemd), shade=True)
    sns.kdeplot(np.array(Bemu), shade=True)
    plt.savefig(os.path.join(out1,'emdist'))
    plt.close()


    plt.hist(np.array(area_fs))
    plt.savefig(os.path.join(out1, 'scadlam'))
    plt.clf()
    plt.hist(np.array(area_fm))
    plt.savefig(os.path.join(out1, 'mcplam'))
    plt.clf()

    plt.scatter(Buf1s, Buf2s, c=colors_ufem, alpha=0.5)
    plt.savefig(os.path.join(out1,'unfused'))
    plt.clf()
    plt.scatter(Bfr1s, Bfr2s, c=colors_ufem, s=area_fr, alpha=0.5)
    plt.savefig(os.path.join(out1,'fused l2'))
    plt.clf()
    plt.scatter(Bfs1s, Bfs2s, c=colors_fs, s=area_fs, alpha=0.5)
    plt.savefig(os.path.join(out1,'scad'))
    plt.clf()
    plt.scatter(Bfm1s, Bfm2s, c=colors_fm, s=area_fm, alpha=0.5)
    plt.savefig(os.path.join(out1,'mcp'))
    plt.clf()
    plt.scatter(Bem1s, Bem2s, c=colors_ufem, alpha=0.5)
    plt.savefig(os.path.join(out1,'em'))

    plt.show()



def test_em():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    N_TF = 20
    N_G = 200
    N_cond = 15
    amt_fused = 1.0
    orth_err = [0,0.3,0.5,0.7,0.9]
    lamS = 1
    k = 10
    if not os.path.exists(os.path.join('data','fake_data','test_em1')):
        os.mkdir(os.path.join('data','fake_data','test_em1'))
    #iterate over how much fusion 
    errors_em = np.zeros((k,0))
    errors_l2 = np.zeros((k,0))
    errors_uf = np.zeros((k,0))
    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','test_em1','dat_'+str(N))
        ds.write_fake_data1(N1 = N_cond*k, N2 = N_cond*k, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, orth_falseneg = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
        lamR = 1
        lamP = 1.0 #priors don't matter
        errd = fg.cv_model2(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct_em', reverse = True, special_args = {'em_it':5, 'f':1, 'uf':1}, cv_both = (True, True))[0]
        errl = fg.cv_model2(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))[0]
        erru = fg.cv_model2(out, lamP=lamP, lamR=lamR, lamS=0, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))[0]
        errors_em = np.hstack((errors_em, errd))
        errors_l2 = np.hstack((errors_l2, errl))
        errors_uf = np.hstack((errors_uf, erru))
    
    errors = np.dstack((errors_em, errors_l2, errors_uf))
    print errors[:,:,0]
    print errors[:,:,1]
    print errors[:,:,2]
    print errors.shape
    step = pd.Series(orth_err)
    solver = pd.Series(["EM", "fused L2", "L2"], name="solver")
    plt.close()
    sns.tsplot(errors, time=step, condition=solver, value="mean squared error")
    plt.savefig(os.path.join(os.path.join('data','fake_data','test_em1','fig5')))
    plt.show()
    return errors

def test_em2():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    N_TF = 5
    N_G = 20
    amt_fused = 1.0
    orth_err = [0,0.3,0.5,0.7,0.9]
    lamS = 1
    k = 10
    if not os.path.exists(os.path.join('data','fake_data','test_em2')):
        os.mkdir(os.path.join('data','fake_data','test_em2'))
    #iterate over how much fusion 
    errors_em = np.zeros((k,0))
    errors_l2 = np.zeros((k,0))
    errors_uf = np.zeros((k,0))
    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','test_em2','dat_'+str(N))
        ds.write_fake_data1(N1 = 10*10, N2 = 10*10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, orth_falseneg = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
        lamR = 0.1
        lamP = 1.0 #priors don't matter
        errd1 = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_em', reverse = True, special_args = {'em_it':5, 'f':1, 'uf':1}, cv_both = (True, True))
        errl1 = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
        erru1 = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=0, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
        errd = fg.cv_model2(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_em', reverse = True, special_args = {'em_it':5, 'f':1, 'uf':1}, cv_both = (True, True))[0]
        errl = fg.cv_model2(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))[0]
        erru = fg.cv_model2(out, lamP=lamP, lamR=lamR, lamS=0, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))[0]
        errors_em = np.hstack((errors_em, errd))
        errors_l2 = np.hstack((errors_l2, errl))
        errors_uf = np.hstack((errors_uf, erru))
    
    errors = np.dstack((errors_em, errors_l2, errors_uf))
    print errors[:,:,0]
    print errors[:,:,1]
    print errors[:,:,2]
    print errors.shape
    step = pd.Series(orth_err)
    solver = pd.Series(["EM", "fused L2", "L2"], name="solver")
    plt.close()
    sns.tsplot(errors, time=step, condition=solver, value="mean squared error")
    plt.savefig(os.path.join(os.path.join('data','fake_data','test_em1','fig4')))
    plt.show()
    return (errors,errd1, errl1, erru1)


def test_em3():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    N_TF = 10
    N_G = 200
    amt_fused = 1.0
    orth_err = [0,0.3,0.5,0.7,0.9]
    lamS = 1.0
    if not os.path.exists(os.path.join('data','fake_data','test_em4')):
        os.mkdir(os.path.join('data','fake_data','test_em4'))
    #iterate over how much fusion
    errors_em = np.zeros(len(orth_err))
    errors_l2 = np.zeros(len(orth_err))
    errors_uf = np.zeros(len(orth_err))
    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','test_em4','dat_'+str(N))
        ds.write_fake_data1(N1 = 10*10, N2 = 10*10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, orth_falseneg = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
        lamR = 0.1
        lamP = 1.0 #priors don't matter
        errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_em', reverse = True, special_args = {'em_it':5, 'f':1, 'uf':1}, cv_both = (True, True))
        errl = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
        erru = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=0, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
#        errors_em[i] = errd['mse'][0]
#        errors_l2[i] = errl['mse'][0]
#        errors_uf[i] = erru['mse']

#    colorlist = [[0,0,1],[0,1,0],[1,0,0],[0.5,0,0.5]]
#    for r, amnt in enumerate(orth_err):
#        plt.plot(orth_err, errors_em[r,:], color = colorlist[r])
#    for r, amnt in enumerate(orth_err):
#        plt.plot(orth_err, errors_l2[r,:], '--', color = colorlist[r])
#    #plt.legend(lamSs+lamSs)
#    plt.xlabel('orth error')
#    plt.ylabel('mean squared error')
#    plt.savefig(os.path.join(os.path.join('data','fake_data','test_em3','fig3')))
#    plt.figure()

def test_mcp_2():
    N_TF = 10
    N_G = 200
    amt_fused = 1.0
    orth_err = [0,0.3,0.5,0.7,0.9]
    lamS = 1.0
    if not os.path.exists(os.path.join('data','fake_data','test_mcp_2')):
        os.mkdir(os.path.join('data','fake_data','test_mcp_2'))
    #iterate over how much fusion
    errors_em = np.zeros(len(orth_err))
    errors_l2 = np.zeros(len(orth_err))
    errors_uf = np.zeros(len(orth_err))
    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','test_mcp_2','dat_'+str(N))
        ds.write_fake_data1(N1 = 10, N2 = 10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, orth_falseneg = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
        lamR = 0.1
        lamP = 1.0 #priors don't matter
        #errmr = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_mcp_r', reverse = True, cv_both = (True, True))
        errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
        errm = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=0, k=10, solver='solve_ortho_direct_mcp', reverse = True, cv_both = (True, True),special_args={'m_it':5, 'a':0.5})

#looks at the distribution of absolute values of correlations for prior interactions, and an equally sized set of random interactions
def test_prior_corr():
    
    bactf = 'data/bacteria_standard'
    ds1 = ds.standard_source(bactf,0)
    #ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_priors()
    
    #(priors2, signs2) = ds2.get_priors()
    #(constraints, marks, orths) = ds.load_constraints(bactf)
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    #(e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
    
    prior_cons = fr.priors_to_constraints([ds1.name], [genes1], [tfs1], priors1, 1.0)
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

#looks at distribution of correlations for repression and activation priors
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


# we look at the TF x Gene correlation matrices in each species, for genes which have orthologies. I'll then order one to be pretty, and order the other in the same way
def check_ortho_corr():
    bactf = 'data/bacteria_standard'
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
    
    
    
#makes a pretty order
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

#look at whether the orthology mapping preserves tfness of genes
#turns out lots of tfs are orthologs of non-tfs, and vice versa
def verify_orth_preserves_tf():
    bactf = 'data/bacteria_standard'
    ds1 = ds.standard_source(bactf,0)
    ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_priors()
    
    (priors2, signs2) = ds2.get_priors()
    (constraints, marks, orths) = ds.load_constraints(bactf)
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
    
    orth_set_subt = set(map(lambda orth: orth.genes[0].name, orths))
    orth_set_anth = set(map(lambda orth: orth.genes[1].name, orths))

    tfs_orth_subt = filter(lambda tf: tf in orth_set_subt, tfs1)
    tfs_orth_anth = filter(lambda tf: tf in orth_set_anth, tfs2)

    gen_orth_subt = filter(lambda g: g in orth_set_subt, genes1)
    gen_orth_anth = filter(lambda g: g in orth_set_anth, genes2)

    gene_inds_subt = {genes1[i] : i for i in range(len(genes1))}
    gene_inds_anth = {genes2[i] : i for i in range(len(genes2))}

    subt_to_anth = {orths[x].genes[0].name : orths[x].genes[1].name for x in range(len(orths))}
    anth_to_subt = {orths[x].genes[1].name : orths[x].genes[0].name for x in range(len(orths))}
    #map subtilis tfs onto their orthologs
    subt_tfs_mapped = map(lambda x: subt_to_anth[x], tfs_orth_subt)
    anth_tfs_mapped = map(lambda x: anth_to_subt[x], tfs_orth_anth)

    print filter(lambda x: not x in tfs2, subt_tfs_mapped)
    print filter(lambda x: not x in tfs1, anth_tfs_mapped)
    print '%d subtilis tfs map to %d anthracis tfs and %d non-tfs' % ( len(tfs1), len(filter(lambda x: x in tfs2, subt_tfs_mapped)), len(filter(lambda x: not x in tfs2, subt_tfs_mapped)))
    print '%d anthracis tfs map to %d subtilis tfs and %d non-tfs' % ( len(tfs2), len(filter(lambda x: x in tfs1, anth_tfs_mapped)), len(filter(lambda x: not x in tfs1, anth_tfs_mapped)))


#looks at distribution of correlations for repression and activation priors in anthracis, after mapping from subtilis
def test_prior_sign_corr_orth():

    bactf = 'data/bacteria_standard'
    ds1 = ds.standard_source(bactf,0)
    ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_priors()
    
    
    (constraints, marks, orths) = ds.load_constraints(bactf)
    print len(constraints)
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()

    subt_to_anth = {orths[x].genes[0].name : orths[x].genes[1].name for x in range(len(orths))}
    
    priors_a = map(lambda ind1: priors1[ind1], filter(lambda ind2: signs1[ind2] == 1, range(len(signs1))))
    priors_r = map(lambda ind1: priors1[ind1], filter(lambda ind2: signs1[ind2] == -1, range(len(signs1))))



    priors_a = filter(lambda p: p[0].name in subt_to_anth and p[1].name in subt_to_anth, priors_a)
    priors_r = filter(lambda p: p[0].name in subt_to_anth and p[1].name in subt_to_anth, priors_r)


    gene_ind_anth = {genes2[i] : i for i in range(len(genes2))}
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

        tf_anth = subt_to_anth[tfs1[tfi]]
        g_anth = subt_to_anth[genes1[gi]]

        tfi_anth = gene_ind_anth[tf_anth]
        gi_anth = gene_ind_anth[g_anth]

        exp_tf = e2_tr[:,[tfi_anth]]
        exp_g = e2_tr[:, [gi_anth]]
        
        corrmat = np.corrcoef(np.hstack((exp_tf, exp_g)).T)
        
        corr = (corrmat[1,0])
        repr_interactions[i] = corr

    for i, prior in enumerate(prior_cons_a):
        tfi = prior.c1.r
        gi = prior.c1.c

        tf_anth = subt_to_anth[tfs1[tfi]]
        g_anth = subt_to_anth[genes1[gi]]

        tfi_anth = gene_ind_anth[tf_anth]
        gi_anth = gene_ind_anth[g_anth]

        exp_tf = e2_tr[:,[tfi_anth]]
        exp_g = e2_tr[:, [gi_anth]]
        
        corrmat = np.corrcoef(np.hstack((exp_tf, exp_g)).T)
        
        corr = (corrmat[1,0])
        acti_interactions[i] = corr

    for i, prior in enumerate(prior_cons_a):
        tfi = np.random.randint(0, len(tfs2))
        gi = np.random.randint(0, len(genes2))
        exp_tf = e2_tr[:,[tfi]]
        exp_g = e2_tr[:, [gi]]
        
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


#makes heatmap showing effect of lamS on networks in L2 fusion, using optimal lamR
def L2fusiontest():    
    N_TF = 20
    N_G = 200
    pct_fused = list(np.arange(0,1.0,0.1))

    lamRs = list(np.arange(0,3.0,0.1))
    lamSs = list(np.arange(0,3.0,0.1))
    lamP = 1.0
    mse_array = np.zeros((len(pct_fused),len(lamSs)))

    out1 = os.path.join('data','fake_data','L2fusiontest')
    k = 5#cv folds
    if not os.path.exists(out1):
        os.mkdir(out1)


    for j, N in enumerate(pct_fused):
        out2 = os.path.join(out1,'dat_'+str(N))
        ds.write_fake_data1(out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1, pct_fused=N)        
        lamP = 1.0 #priors don't matter
        for i, fused in enumerate(pct_fused):
            for j, lamS in enumerate(lamSs):
                (mse, best_lamP, best_lamR, best_lamS) = fg.grid_search_params(out2, lamP=[lamP], lamR=lamRs, lamS=[lamS], k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
                mse_array[i,j] += mse

    mse_df = pd.DataFrame(mse_array, index=pct_fused, columns=lamSs)
    sns.heatmap(mse_df)
    plt.show()




#this looks at the distributions Betas for +/- priors, and an equal number of non-priors in subtilis
def test_prior_sign_betas(lamP=1.0, lamR=1.0,lamS=0):
    #first, get the betas. don't fuse
    (B0, _) = test_bacteria(lamP, lamR, lamS)
    print 'hallo'
    bactf = 'data/bacteria_standard'
    
    ds1 = ds.standard_source(bactf,0)
    #ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_priors()
    
    #(priors2, signs2) = ds2.get_priors()
    #(constraints, marks, orths) = ds.load_constraints(bactf)
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    #(e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()
    aupr = fg.eval_network_pr(B0, genes1, tfs1, priors1, exclude_tfs=False)
    auc = fg.eval_network_roc(B0, genes1, tfs1, priors1, exclude_tfs=False)

    print 'performance: aupr %f, auc %f' % (aupr, auc)
    priors_a = map(lambda ind1: priors1[ind1], filter(lambda ind2: signs1[ind2] == 1, range(len(signs1))))

    priors_r = map(lambda ind1: priors1[ind1], filter(lambda ind2: signs1[ind2] == -1, range(len(signs1))))

    prior_cons_r = fr.priors_to_constraints([ds1.name], [genes1], [tfs1], priors_r, 1.0)
    prior_cons_a = fr.priors_to_constraints([ds1.name], [genes1], [tfs1], priors_a, 1.0)
    #gene_corr = np.corrcoef(e1_tr.T)
    #we now rely on the fact that tfs occur before genes, and in the same order as they do in the list of genes, so indices are the same
    
    repr_interactions = []
    acti_interactions = []
    rand_interactions = []
    for i, prior in enumerate(prior_cons_r):
        tfi = prior.c1.r
        gi = prior.c1.c
        exp_tf = e1_tr[:,[tfi]]
        exp_g = e1_tr[:, [gi]]
        if tfi < B0.shape[0]:
            repr_interactions.append(B0[tfi, gi])

    for i, prior in enumerate(prior_cons_a):
        tfi = prior.c1.r
        gi = prior.c1.c
        exp_tf = e1_tr[:,[tfi]]
        exp_g = e1_tr[:, [gi]]        
        if tfi < B0.shape[0]:
            acti_interactions.append(B0[tfi, gi])

    for i, prior in enumerate(prior_cons_a):
        tfi = np.random.randint(0, len(tfs1))
        gi = np.random.randint(0, len(genes1))
        exp_tf = e1_tr[:,[tfi]]
        exp_g = e1_tr[:, [gi]]
        if tfi < B0.shape[0]:
            rand_interactions.append(B0[tfi, gi])
    repr_interactions = 1000*np.array(repr_interactions)
    acti_interactions = 1000*np.array(acti_interactions)
    rand_interactions = 1000*np.array(rand_interactions)
    sns.kdeplot(repr_interactions, shade=True, label = 'repression, %f'% np.mean(repr_interactions))
    plt.hold(True)
    sns.kdeplot(acti_interactions, shade=True, label = 'activation, %f'% np.mean(acti_interactions))
    sns.kdeplot(rand_interactions, shade=True, label = 'non-priors, %f' % np.mean(rand_interactions))
    
    plt.xlabel('beta')
    plt.legend()
    plt.ylabel('frequency')    
    
    plt.show()

    sns.kdeplot(np.abs(np.hstack((repr_interactions, acti_interactions))), label='abs priors %f' % (0.5*np.mean(np.abs(repr_interactions)) + 0.5*np.mean(np.abs(acti_interactions))))
    sns.kdeplot(np.abs(np.hstack((rand_interactions, rand_interactions))), label='abs non_priors %f' % (np.mean(np.abs(rand_interactions))))

    plt.show()
#looks at distribution of correlations for repression and activation priors in anthracis, after mapping from subtilis
def test_prior_sign_betas_orth(lamP=1.0, lamR=1.0,lamS=0):
    
    bactf = 'data/bacteria_standard'
    ds1 = ds.standard_source(bactf,0)
    ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_priors()
    
    
    (constraints, marks, orths) = ds.load_constraints(bactf)
    
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()

    subt_to_anth = {orths[x].genes[0].name : orths[x].genes[1].name for x in range(len(orths))}
    
    priors_a = map(lambda ind1: priors1[ind1], filter(lambda ind2: signs1[ind2] == 1, range(len(signs1))))
    priors_r = map(lambda ind1: priors1[ind1], filter(lambda ind2: signs1[ind2] == -1, range(len(signs1))))

    priors_a = filter(lambda p: p[0].name in subt_to_anth and p[1].name in subt_to_anth, priors_a)
    priors_r = filter(lambda p: p[0].name in subt_to_anth and p[1].name in subt_to_anth, priors_r)


    gene_ind_anth = {genes2[i] : i for i in range(len(genes2))}
    prior_cons_r = fr.priors_to_constraints([ds1.name], [genes1], [tfs1], priors_r, 1.0)
    prior_cons_a = fr.priors_to_constraints([ds1.name], [genes1], [tfs1], priors_a, 1.0)
    #gene_corr = np.corrcoef(e1_tr.T)
    #we now rely on the fact that tfs occur before genes, and in the same order as they do in the list of genes, so indices are the same
    
    repr_interactions = []
    acti_interactions = []
    rand_interactions = []
    
    (B0, B1) = test_bacteria(lamP, lamR, lamS)
    aupr = fg.eval_network_pr(B0, genes1, tfs1, priors1, exclude_tfs=False)
    auc = fg.eval_network_roc(B0, genes1, tfs1, priors1, exclude_tfs=False)

    print 'performance: aupr %f, auc %f' % (aupr, auc)

    for i, prior in enumerate(prior_cons_r):
        tfi = prior.c1.r
        gi = prior.c1.c

        tf_anth = subt_to_anth[tfs1[tfi]]
        g_anth = subt_to_anth[genes1[gi]]

        tfi_anth = gene_ind_anth[tf_anth]
        gi_anth = gene_ind_anth[g_anth]
        if tfi_anth < B1.shape[0]:
            repr_interactions.append(B1[tfi_anth, gi_anth])


    for i, prior in enumerate(prior_cons_a):
        tfi = prior.c1.r
        gi = prior.c1.c

        tf_anth = subt_to_anth[tfs1[tfi]]
        g_anth = subt_to_anth[genes1[gi]]

        tfi_anth = gene_ind_anth[tf_anth]
        gi_anth = gene_ind_anth[g_anth]
        if tfi_anth < B1.shape[0]:
            acti_interactions.append(B1[tfi_anth, gi_anth])


    for i, prior in enumerate(prior_cons_a):
        tfi = np.random.randint(0, len(tfs2))
        gi = np.random.randint(0, len(genes2))
        exp_tf = e2_tr[:,[tfi]]
        exp_g = e2_tr[:, [gi]]
        
        rand_interactions.append(B1[tfi, gi])

    repr_interactions = 1000*np.array(repr_interactions)
    acti_interactions = 1000*np.array(acti_interactions)
    rand_interactions = 1000*np.array(rand_interactions)
    
    sns.kdeplot(repr_interactions, shade=True, label = 'repression, %f'% np.mean(repr_interactions))
    plt.hold(True)
    sns.kdeplot(acti_interactions, shade=True, label = 'activation, %f'% np.mean(acti_interactions))
    sns.kdeplot(rand_interactions, shade=True, label = 'non-priors, %f' % np.mean(rand_interactions))
    
    plt.xlabel('beta')
    plt.legend()
    plt.ylabel('frequency')    
    
    plt.show()


    sns.kdeplot(np.abs(np.hstack((repr_interactions, acti_interactions))), label='abs priors %f' % (0.5*np.mean(np.abs(repr_interactions)) + 0.5*np.mean(np.abs(acti_interactions))))
    sns.kdeplot(np.abs(np.hstack((rand_interactions, rand_interactions))), label='abs non_priors %f' % (np.mean(np.abs(rand_interactions))))

    plt.show()

#this function fits the anthracis network, then maps interactions to the subtilis network, and evaluates aupr/auc
def eval_mapped_performance(lamP=1.0, lamR=1.0,lamS=0):
    
    bactf = 'data/bacteria_standard'
    ds1 = ds.standard_source(bactf,0)
    ds2 = ds.standard_source(bactf,1)
    (priors1, signs1) = ds1.get_priors()
    
    
    (constraints, marks, orths) = ds.load_constraints(bactf)
    
    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()

    subt_to_anth = {orths[x].genes[0].name : orths[x].genes[1].name for x in range(len(orths))}
    anthracis_gene_inds = {genes2[i] : i for i in range(len(genes2))}
    subtilis_gene_inds = {genes1[i] : i for i in range(len(genes1))}
    
    
    (B0, B1) = test_bacteria(lamP, lamR, lamS)
    
    aupr = fg.eval_network_pr(B0, genes1, tfs1, priors1, exclude_tfs=False)
    auc = fg.eval_network_roc(B0, genes1, tfs1, priors1, exclude_tfs=False)

    print 'performance: aupr %f, auc %f' % (aupr, auc)

    aupr = fg.eval_network_pr(B0, genes1, tfs1, priors1, exclude_tfs=False, constraints=constraints, sub=0)
    auc = fg.eval_network_roc(B0, genes1, tfs1, priors1, exclude_tfs=False, constraints=constraints, sub=0)

    print 'performance-constr: aupr %f, auc %f' % (aupr, auc)

    B0_mapped = np.zeros(B0.shape)
    B0_mapped2 = np.zeros(B0.shape) #same thing but using constraints
    filled = 0
    for gi, gene in enumerate(genes1):
        for tfi, tf in enumerate(tfs1):
            if gene in subt_to_anth and tf in subt_to_anth:
                gi_mapped = anthracis_gene_inds[subt_to_anth[gene]]
                tfi_mapped = anthracis_gene_inds[subt_to_anth[tf]]
                if tfi_mapped < B1.shape[0]: #exclude possibility of a tf <-> non-tf mapping
                    B0_mapped[tfi, gi] = B1[tfi_mapped, gi_mapped]
                    filled += 1
    
    print 'filled in %d' % filled
    
    for con in constraints:
        if con.c1.sub == 0:
            con_fr_g = genes1[con.c1.c]
            con_fr_tf = genes1[con.c1.r]
            con_to_g = genes2[con.c2.c]
            con_to_tf = genes2[con.c2.r]
            
            con_to_m_g = subt_to_anth[con_fr_g]
            con_to_m_tf = subt_to_anth[con_fr_tf]

            if con_to_g != con_to_m_g or con_to_tf != con_to_m_tf:
                print 'constraint maps (%s, %s) to (%s, %s). Orthology maps (%s, %s) to (%s, %s). '%(con_fr_g, con_fr_tf,con_to_g,con_to_tf,con_fr_g, con_fr_tf,con_to_m_g,con_to_m_tf,)
            B0_mapped2[con.c1.r, con.c1.c] = B1[con.c2.r, con.c2.c]
            #double check this is the same as B0_mapped
            if B0_mapped[con.c1.r, con.c1.c] != B0_mapped2[con.c1.r, con.c1.c]:
                print 'wtf???'
                print 'constraint maps (%s, %s) to (%s, %s). Orthology maps (%s, %s) to (%s, %s). '%(con_fr_g, con_fr_tf,con_to_g,con_to_tf,con_fr_g, con_fr_tf,con_to_m_g,con_to_m_tf,)
                print B0_mapped[con.c1.r, con.c1.c]
                print B0_mapped2[con.c1.r, con.c1.c]
                print 'cons (%d, %d) to (%d, %d)' % (con.c1.r, con.c1.c, con.c2.r, con.c2.c)
                print 'omap (%d, %d) to (%d, %d)' % (subtilis_gene_inds[con_fr_tf], subtilis_gene_inds[con_fr_g], anthracis_gene_inds[con_to_tf], anthracis_gene_inds[con_to_g])
    print 'there are %d different entries' % (B0_mapped != B0_mapped2).sum()
    aupr = fg.eval_network_pr(B0_mapped, genes1, tfs1, priors1, exclude_tfs=False)
    auc = fg.eval_network_roc(B0_mapped, genes1, tfs1, priors1, exclude_tfs=False)

    print 'performance, mapped: aupr %f, auc %f' % (aupr, auc)
    

    aupr = fg.eval_network_pr(B0_mapped2, genes1, tfs1, priors1, exclude_tfs=False, constraints=constraints, sub=0)
    auc = fg.eval_network_roc(B0_mapped2, genes1, tfs1, priors1, exclude_tfs=False, constraints=constraints, sub=0)

    print 'performance-constr, mapped2: aupr %f, auc %f' % (aupr, auc)





def plot_bacteria_performance(lamP=1.0, lamR=5, lamSs=[0], k=20):
    import pickle
    out = 'data/bacteria_standard'
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamSs))) for m in metrics} #anthracis

    
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model3(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False)
        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]

    to_plot = np.dstack((err_dict1['aupr'], err_dict1['aupr_con']))
    linedesc = pd.Series(['full','constrained'],name='error type')
    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='aupr')
    with file('err_dict1','w') as f:
        pickle.dump(err_dict1, f)
    
    plt.show()


#plots error dictionaries
#this is really for debugging plot_bacteria_performance 

def plot_synthetic_performance(lamP=1.0, lamR=5, lamSs=[0], k=20):
    N = 10
    N_TF = 20
    N_G = 30
    out = os.path.join('data','fake_data','plot_synthetic_performance2')
    ds.write_fake_data1(N1 = k*N, N2 = 5*k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.8, sparse=0.5, fuse_std = 0.0, orth_falsepos=0.0,orth_falseneg=0.0)
    
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamSs))) for m in metrics} #anthracis

    
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model3(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False)
        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]

    to_plot = np.dstack((err_dict1['aupr'], err_dict1['aupr_con']))
    linedesc = pd.Series(['full','constrained'],name='error type')
    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='aupr')
    
    plt.show()


#plots error dictionaries
#this one includes lots of incorrect orthology
def plot_synthetic_performance2(lamP=1.0, lamR=5, lamSs=[0], k=20):
    N = 10
    N_TF = 20
    N_G = 30
    out = os.path.join('data','fake_data','plot_synthetic_performance2')
    ds.write_fake_data1(N1 = k*N, N2 = 5*k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.8, sparse=0.5, fuse_std = 0.0, orth_falsepos=0.5,orth_falseneg=0.5)
    
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamSs))) for m in metrics} #anthracis

    
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model3(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False)
        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]

    to_plot = np.dstack((err_dict1['aupr'], err_dict1['aupr_con']))
    linedesc = pd.Series(['full','constrained'],name='error type')
    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='aupr')
    
    plt.show()


#returns error dictionaries on a simple synthetic dataset across a range of parameter combinations. This function uses the cv code that evaluates every model on each individual fold
def synthetic_performance_synch(lamPs=[1.0], lamRs=[5], lamSs=[0], k=20):
    N = 10
    N_TF = 20
    N_G = 30
    out = os.path.join('data','fake_data','plot_synthetic_performance_synch')
    ds.write_fake_data1(N1 = k*N, N2 = 5*k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.8, sparse=0.5, fuse_std = 0.0, orth_falsepos=0.45,orth_falseneg=0.45)
    
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    
    (errd1, errd2) = fg.cv_model_params(out, lamPs=lamPs, lamRs=lamRs, lamSs=lamSs, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False)
    plot_errd_synch(errd1, lamPs, lamRs, lamSs, primary_ax=2, metric='aupr')
    plt.figure()
    plot_errd_synch(errd1, lamPs, lamRs, lamSs, primary_ax=1, metric='aupr')
    plt.show()
    return (errd1, errd2)

#returns (and saves) error across a range of parameter combinations on real data. This function uses the cv code that evaluates every model on each individual fold
def bacteria_performance_synch(lamPs=[1.0], lamRs=[5], lamSs=[0], k=20):
    out = os.path.join('data','bacteria_standard')
    
    
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    
    (errd1, errd2) = fg.cv_model_params(out, lamPs=lamPs, lamRs=lamRs, lamSs=lamSs, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False)
    import pickle
    with file('picklejar','w') as f:
        pickle.dump( (lamPs, lamRs, lamSs, k, errd1, errd2), f)

#function for making reasonably nice looking plots of error dictionaries 
def plot_errd_synch(errd, lamPs, lamRs, lamSs, primary_ax, metric):
    parameters = (lamPs, lamRs, lamSs)
    #axis 0 is always cv folds
    lines = []
    plot_acc = None
    line_descs = []
    pnames = ['lamP','lamR','lamS']
    for Pi in range(len(lamPs)):
        for Ri in range(len(lamRs)):
            for Si in range(len(lamSs)):
                sl_inds = [Pi, Ri, Si]
                if sl_inds[primary_ax] != 0:
                    continue
                
                sl_inds[primary_ax] = range(len(parameters[primary_ax]))
                
                to_plot = errd[metric][:, sl_inds[0], sl_inds[1], sl_inds[2]]
                if plot_acc == None:
                    plot_acc = np.dstack((to_plot, ))
                else:
                    plot_acc = np.dstack((plot_acc, to_plot))
                line_desc = []
                
                for slindi in range(len(sl_inds)):
                    slind = sl_inds[slindi]
                    if slindi != primary_ax: #only use the individual values
                        line_desc.append(pnames[slindi] + '=%f' % parameters[slindi][slind])
                line_descs.append('\n'.join(line_desc))

    
    
    linedesc = pd.Series(line_descs,name='params')
    
    xs = pd.Series(parameters[primary_ax], name=pnames[primary_ax])
    
    plot_slice = plot_acc[:,:,:]
    
    sns.tsplot(plot_slice, time=xs,condition=linedesc, value='aupr')
    
    
    
