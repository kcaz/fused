import fused_reg as fr
import data_sources as ds
import fit_grn as fg
import numpy as np
import os
from matplotlib import pyplot as plt
#This file is just a list of experiments. Only code that is used nowhere else is appropriate (ie general plotting code should go somewhere else


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
    plt.savefig(os.path.join(os.path.join('data','fake_data','sanity1','fig1')))
    plt.figure()
    plt.plot(errors)
    plt.legend(lamSs)
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



def test_scad():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    N_TF = 10
    N_G = 200
    amt_fused = 1.0
    orth_err = [0,0.3,0.5,1.0]
    lamSs = [0, 0.5, 0.5, 1, 5]
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
    for c, amnt in enumerate(orth_err):
        plt.plot(orth_err, errors_scad[:,c], color = colorlist[c])
    for c, amnt in enumerate(orth_err):
        plt.plot(orth_err, errors_l2[:,c], '--', color = colorlist[c])
    plt.legend(lamSs+lamSs)
    plt.xlabel('orth error')
    plt.ylabel('mean squared error')
    plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad','fig2')))
    plt.figure()


def test_scad2():
#create simulated data set with false orthology and run fused scad + visualize scad penalty at each s_it
     
    N_TF = 10
    N_G = 200
    amt_fused = 1.0
    orth_err = 0.3
    lamS = 0.5

    if not os.path.exists(os.path.join('data','fake_data','test_scad')):
        os.mkdir(os.path.join('data','fake_data','test_scad'))

    out = os.path.join('data','fake_data','test_scad','ortherr'+str(orth_err))
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
        new_fuse_constraints = fr.scad(Bs_unfused, fuse_constraints, lamS, lamW=None, a=0.5)

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
        plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad',str(fold))))
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
    #ds.write_fake_data1(out_dir=out, N1=N,N2=N,tfg_count1=(2,1),tfg_count2=(2,1),sparse=0.0,fuse_std=0.0,measure_noise1=0.3,measure_noise2=0.3)
    

    (br1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (br2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
    plt.plot(br1[0,2], br1[1,2], '*',c=[0.5,0,0.5],markersize=30)
    
    plt.plot(br2[0,2], br2[1,2], '*',c=[0.5,0,0.5],markersize=30)
    
    for lamS in lamSs:
    
    #fg.cv_model1(data_fn = out, lamP=lamPs[0], lamR=lamRs[0], lamS=lamSs[i], k= 10)
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS)
        plt.plot(b1[0,2], b1[1,2], 'or',markersize=20*lamS)
        plt.plot(b2[0,2], b2[1,2], 'ob',markersize=20*lamS)
        #print b1
        #print br1

    
    plt.rcParams.update({'font.size': 18})
    
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
    #ds.write_fake_data1(out_dir=out, N1=N,N2=N,tfg_count1=(2,1),tfg_count2=(2,1),sparse=0.0,fuse_std=0.0,measure_noise1=0.3,measure_noise2=0.3, pct_fused=0.5, orth_falsepos=0.99)#think there are orths that are not
    

    (br1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (br2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
    plt.plot(br1[0,2], br1[1,2], '*',c='r',markersize=30)
    
    plt.plot(br2[0,2], br2[1,2], '*',c='b',markersize=30)
    
    for lamS in lamSs:
    
        #fg.cv_model1(data_fn = out, lamP=lamPs[0], lamR=lamRs[0], lamS=lamSs[i], k= 10)
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS)
        plt.plot(b1[0,2], b1[1,2], 'or',markersize=20*lamS)
        plt.plot(b2[0,2], b2[1,2], 'ob',markersize=20*lamS)
        #print b1
        #print br1

    
    plt.rcParams.update({'font.size': 18})
    
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
    #ds.write_fake_data1(out_dir=out, N1=N1,N2=N2,tfg_count1=(2,1),tfg_count2=(2,1),sparse=0.0,fuse_std=0.0,measure_noise1=0.3,measure_noise2=0.3)
    

    (br1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (br2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
    plt.plot(br1[0,2], br1[1,2], '*',c=[0.5,0,0.5],markersize=30)
    
    plt.plot(br2[0,2], br2[1,2], '*',c=[0.5,0,0.5],markersize=30)
    
    for lamS in lamSs:
    
    #fg.cv_model1(data_fn = out, lamP=lamPs[0], lamR=lamRs[0], lamS=lamSs[i], k= 10)
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS)
        plt.plot(b1[0,2], b1[1,2], 'or',markersize=20*lamS)
        plt.plot(b2[0,2], b2[1,2], 'ob',markersize=20*lamS)
        #print b1
        #print br1

    
    plt.rcParams.update({'font.size': 18})
    
    plt.xlabel('coefficient1')
    plt.ylabel('coefficient2')
    plt.show()


#tests fusion visually using a pair of similar two-coefficient networks
#this network only has 50% ortho coverage
#NOW USES SCAD!
def test_2coeff_fuse_HS():
    lamPs = np.array([1])
    lamRs = np.array([0.1])
    lamSs = np.linspace(0,1,10)
    
    out = os.path.join('data','fake_data','2coeff_fuse_HS')
    if not os.path.exists(out):
        os.mkdir(out)

    N = 30
    ds.write_fake_data1(out_dir=out, N1=N,N2=N,tfg_count1=(2,1),tfg_count2=(2,1),sparse=0.0,fuse_std=0.0,measure_noise1=0.0,measure_noise2=0.0, pct_fused=0.5, orth_falsepos=0.99)#think there are orths that are not
    

    (br1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (br2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
    plt.plot(br1[0,2], br1[1,2], '*',c='r',markersize=30)
    
    plt.plot(br2[0,2], br2[1,2], '*',c='b',markersize=30)
    #we want to grab the fusion constraints that affect the coefficients we care about, specifically 0->2 <--> 0'->2', 1->2 <--> 1' -> 2'
    def assemble_orths(orths):
        for orth in orths:
            print '%d, %d fused to %d, %d - lamS = %f' %(orth.c1.r, orth.c1.c, orth.c2.r, orth.c2.c, orth.lam)
    for lamS in lamSs:
        print 'SOLVING'
        #fg.cv_model1(data_fn = out, lamP=lamPs[0], lamR=lamRs[0], lamS=lamSs[i], k= 10)
        special_args = {'s_it':5, 'orths':None, 'a':1.50}
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS, solver='solve_ortho_direct_scad',special_args = special_args)
        plt.plot(b1[0,2], b1[1,2], 'or',markersize=10*lamS)
        plt.plot(b2[0,2], b2[1,2], 'ob',markersize=10*lamS)
        
        assemble_orths(special_args['orths'])
        #print b1
        #print br1
        
    
    plt.rcParams.update({'font.size': 18})
    plt.axis('equal')
    plt.xlabel('coefficient1')
    plt.ylabel('coefficient2')
    plt.show()
