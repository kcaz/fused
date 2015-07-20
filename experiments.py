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
import scipy.stats
sns.set(palette="Set2")
#This file is just a list of experiments. Only code that is used nowhere else is appropriate (ie general plotting code should go somewhere else

#this is basic - loads all the data, fits the basic model, and returns B
def test_bacteria(lamP, lamR, lamS):
    subt = ds.standard_source('data/bacteria_standard',0)
    anthr = ds.standard_source('data/bacteria_standard',1)
    subt_eu = ds.standard_source('data/bacteria_standard',2)
    
    (bs_e, bs_t, bs_genes, bs_tfs) = subt.load_data()
    (ba_e, ba_t, ba_genes, ba_tfs) = anthr.load_data()

    (bs_priors, bs_sign) = subt.get_priors()
    (ba_priors, ba_sign) = anthr.get_priors()

    Xs = [bs_t, ba_t]
    Ys = [bs_e, ba_e]
    genes = [bs_genes, ba_genes]
    tfs = [bs_tfs, ba_tfs]
    priors = bs_priors + ba_priors
    orth = ds.load_orth('data/bacteria_standard/orth',organisms = [subt.name, anthr.name, subt_eu.name], orgs = [subt.name, anthr.name])
    organisms = [subt.name, anthr.name]

    Bs = fr.solve_ortho_direct(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS)

    return Bs

#load all the anthracis data and some of the subtilis data
def test_bacteria_subs_subt(lamP, lamRs, lamSs, k=20, eval_con=False, pct_priors=0, seed=None):
    out = 'data/bacteria_standard'
    
    errds = []
    for lamR in lamRs:
        for lamS in lamSs:
            (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False, pct_priors=pct_priors, seed=seed)
            
            errds.append((errd1, errd2))
    return errds




#In this experiment, we generate several data sets with a moderate number of TFs and genes. Fusion is total and noiseless. Measure performance as a function of lamS, and performance as a function of the amount of data. Plot both. This is a basic sanity check for fusion helping
def sanity1():
    repeats = 1
    N_TF = 15
    N_G = 100
    data_amnts = [10,30,50,70,90]
    lamSs = [0, 1]
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
                (errd1,errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
                errors[i, j] += errd1['mse'].mean()
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
    N_TF = 20
    N_G = 200
    fixed_data_amt = 20
    data_amnts = np.linspace(5,200,20)
    lamSs = [0]#[0, 0.25,0.5,0.75,1]
    seed = 10
    if not os.path.exists(os.path.join('data','fake_data','increasedata')):
        os.mkdir(os.path.join('data','fake_data','increasedata'))
    #iterate over how much data to use
    errors = np.zeros((len(data_amnts), len(lamSs)))
    for i, N in enumerate(data_amnts):
        out = os.path.join('data','fake_data','increasedata','dat_'+str(N))
        ds.write_fake_data1(N1 = 10*fixed_data_amt, N2 = 10*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = 1.0, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.75, fuse_std = 0.0)
        lamR = 2
        lamP = 1.0 #priors don't matter
        for j, lamS in enumerate(lamSs):
            (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, seed=seed, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
            errors[i, j] += errd2['aupr'].mean()

    adj_error = np.zeros((errors.shape[0],  errors.shape[1]))
    for val in range(errors.shape[0]):
        adj_error[val, 0] = errors[val,0]/errors[0,0]

    return adj_error
    for c, lamS in enumerate(lamSs):
        plt.plot(data_amnts, errors[:,c])

    plt.legend(data_amnts)
    plt.show()
    plt.savefig(os.path.join(os.path.join('data','fake_data','increasedata','fig1')))
#    plt.figure()
#    for c, lamS in enumerate(lamSs):
#        plt.plot(lamSs, errors[:, c])
#    plt.legend(lamSs)
#    plt.savefig(os.path.join(os.path.join('data','fake_data','increasedata','fig2')))
    #plt.show()


def test_scad():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    N_TF = 10
    N_G = 200
    amt_fused = 0.5
    orth_err = [0.25,0.5,0.75,1]
    lamSs = [0, 2, 4]
    if not os.path.exists(os.path.join('data','fake_data','test_scad')):
        os.mkdir(os.path.join('data','fake_data','test_scad'))
    #iterate over how much fusion
    errors_scad = np.zeros((len(orth_err), len(lamSs)))
    errors_l2 = np.zeros((len(orth_err), len(lamSs)))
    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','test_scad','dat_'+str(N))
        ds.write_fake_data1(N1 = 10*10, N2 = 50*10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1)
        lamR = 2
        lamP = 1.0 #priors don't matter
        for j, lamS in enumerate(lamSs):
            settings = fr.get_settings({'s_it':50, 'a':0.6})
            (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_scad', reverse = False, settings = settings, cv_both = (True, True))

            (errd1l, errd2l) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))

            errors_scad[i,j] = errd1['aupr'].mean()
            errors_l2[i,j] = errd1l['aupr'].mean()

    return (errors_scad, errors_l2)
    orth_error = np.array(orth_err)
    colorlist = [[0,0,1],[0,1,0],[1,0,0],[0.5,0,0.5]]
    for r, amnt in enumerate(orth_err):
        plt.plot(orth_error, errors_scad[:,r], color = colorlist[r])
    for r, amnt in enumerate(orth_err):
        plt.plot(orth_error, errors_l2[:,r], '--', color = colorlist[r])
    #plt.legend(lamSs+lamSs)
    plt.xlabel('orth error')
    plt.ylabel('mean squared error')
    plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad','fig3')))
    plt.figure()

def test_scad_opt_params():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    N_TF = 10
    N_G = 200
    amt_fused = 0.5
    orth_err = [0.25,0.5,0.75,1]
    lamSs = [0, 2, 4, 6]
    reps = 30
    a = np.arange(0.1,1,0.1)

    if not os.path.exists(os.path.join('data','fake_data','test_scad')):
        os.mkdir(os.path.join('data','fake_data','test_scad'))
    #iterate over how much fusion
    errors_scad = np.zeros((len(orth_err), len(lamSs), len(a)))
    errors_l2 = np.zeros((len(orth_err), len(lamSs)))
    errors_scad_var = np.zeros((len(orth_err), len(lamSs), len(a)))
    errors_l2_var = np.zeros((len(orth_err), len(lamSs)))

    for p in range(reps):
        for i, N in enumerate(orth_err):
            out = os.path.join('data','fake_data','test_scad','dat_'+str(N)+str(p))
            ds.write_fake_data1(N1 = 20, N2 = 20, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
            lamR = 5
            lamP = 1.0 #priors don't matter
            scad_settings = fr.get_settings({'s_it':50, 'a':ap})
            for j, lamS in enumerate(lamSs):
                (errd1l, errd2l) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
                errors_l2[i,j] += errd1l['mse'].mean()
                errors_l2_var[i,j] += errd1l['mse'].mean()**2
                for k, ap in enumerate(a):
                    (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_scad', reverse = False, settings = scad_settings, cv_both = (True, True))
                    errors_scad[i,j,k] += errd1['mse'].mean()
                    errors_scad_var[i,j,k] += errd1['mse'].mean()**2

    errors_scad /= p
    errors_l2 /= p
    errors_scad_var /= p
    errors_scad_var -= errors_scad
    errors_l2_var /= p
    errors_l2_var -= errors_l2

    return (errors_scad, errors_scad_var, errors_l2, errors_l2_var)
    orth_error = np.array(orth_err)
    colorlist = [[0,0,1],[0,1,0],[1,0,0],[0.5,0,0.5]]
    for r, amnt in enumerate(orth_err):
        plt.plot(orth_error, errors_scad[:,r], color = colorlist[r])
    for r, amnt in enumerate(orth_err):
        plt.plot(orth_error, errors_l2[:,r], '--', color = colorlist[r])
    #plt.legend(lamSs+lamSs)
    plt.xlabel('orth error')
    plt.ylabel('mean squared error')
    plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad','fig3')))
    plt.figure()

def test_scad_opt_params2():
#create simulated data set with false orthology and run fused L2 and fused scad; faster than test_scad_opt_params because doesn't search through a
     
    N_TF = 20
    N_G = 200
    amt_fused = 0.5
    orth_err = [0,0.25,0.5,0.75,1]
    lamSs = [0,2,4,6]#[0, 2, 4, 6]
    seed = 10
    reps = 2

    if not os.path.exists(os.path.join('data','fake_data','test_scad_opt_params2')):
        os.mkdir(os.path.join('data','fake_data','test_scad_opt_params2'))
    #iterate over how much fusion
    errors_scad = np.zeros((len(orth_err), len(lamSs)))
    errors_l2 = np.zeros((len(orth_err), len(lamSs)))
    errors_scad_var = np.zeros((len(orth_err), len(lamSs)))
    errors_l2_var = np.zeros((len(orth_err), len(lamSs)))

    errors_scad2 = np.zeros((len(orth_err), len(lamSs)))
    errors_l22 = np.zeros((len(orth_err), len(lamSs)))
    errors_scad_var2 = np.zeros((len(orth_err), len(lamSs)))
    errors_l2_var2 = np.zeros((len(orth_err), len(lamSs)))


    for p in range(reps):
        for i, N in enumerate(orth_err):
            out = os.path.join('data','fake_data','test_scad_opt_params2','dat_'+str(N))
            ds.write_fake_data1(N1 = 10*5, N2 = 10*50, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1)
            lamR = 2
            lamP = 1.0 #priors don't matter

            for j, lamS in enumerate(lamSs):
                print N
                print lamS
                (errd1l, errd2l) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
                errors_l2[i,j] += errd1l['aupr'].mean()
                errors_l2_var[i,j] += errd1l['aupr'].mean()**2
                print errd1l['aupr'].mean()
                errors_l22[i,j] += errd2l['aupr'].mean()
                errors_l2_var2[i,j] += errd2l['aupr'].mean()**2


                scad_settings = fr.get_settings({'s_it':50, 'per':((1/(1+float(N)))*100)})
                (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_scad', reverse = False, settings = scad_settings, cv_both = (True, True))
                #print 'orth err'
                #print N
                #print lamS
                #plot_fuse_lams(out, scad_settings['cons'])
                print errd1['aupr'].mean()
                errors_scad[i,j] += errd1['aupr'].mean()
                errors_scad_var[i,j] += errd1['aupr'].mean()**2

                errors_scad2[i,j] += errd2['aupr'].mean()
                errors_scad_var2[i,j] += errd2['aupr'].mean()**2

    errors_scad /= reps
    errors_scad2 /=reps
    errors_l2 /= reps
    errors_l22 /= reps
    errors_scad_var /= reps
    errors_scad_var2 /= reps
    errors_scad_var -= errors_scad**2
    errors_scad_var2 -= errors_scad2**2
    errors_l2_var /= reps
    errors_l2_var2 /= reps
    errors_l2_var -= errors_l2**2
    errors_l2_var2 -= errors_l22**2

    return (errors_scad, errors_scad2, errors_scad_var, errors_scad_var2, errors_l2, errors_l22, errors_l2_var, errors_l2_var2)

def test_scad_opt_params3():
#create simulated data set with false orthology and run fused L2 and fused scad; faster than test_scad_opt_params because doesn't search through a
    import copy    

    N_TF = 20
    N_G = 200
    amt_fused = 0.5
    orth_err = [0,0.5,0.75,1]
    lamSs = [0,2,4]#[0, 2, 4, 6]
    seed = 10
    reps = 1
    k = 5

    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con','B_mse']
    err_dict_s = {m : np.zeros((k, len(lamSs))) for m in metrics}
    err_dict_l = {m : np.zeros((k, len(lamSs))) for m in metrics}
    err_s = {o : copy.deepcopy(err_dict_s) for o in orth_err}
    err_l = {o : copy.deepcopy(err_dict_l) for o in orth_err}

    if not os.path.exists(os.path.join('data','fake_data','test_scad_opt_params3')):
        os.mkdir(os.path.join('data','fake_data','test_scad_opt_params3'))
    #iterate over how much fusion

    for p in range(reps):
        for i, N in enumerate(orth_err):
            out = os.path.join('data','fake_data','test_scad_opt_params3','dat_'+str(N))
            ds.write_fake_data1(N1 = 5*5, N2 = 5*50, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1)
            lamR = 2
            lamP = 1.0 #priors don't matter

            for j, lamS in enumerate(lamSs):
                print N
                print lamS
                (errd1l, errd2l) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=5, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
                print errd1l['aupr']
                for metric in metrics:
                    err_l[N][metric][:, [j]] = errd1l[metric]
                print err_l[N]['aupr']

                scad_settings = fr.get_settings({'s_it':50, 'per':((1/(1+float(N)))*100)})
                (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=5, solver='solve_ortho_direct_scad', reverse = False, settings = scad_settings, cv_both = (True, True))
                print errd1['aupr']
                for metric in metrics:
                    err_s[N][metric][:, [j]] = errd1[metric]
                print err_s[N]['aupr']

    o0 = np.dstack((err_l[0]['aupr'], err_s[0]['aupr']))
    o05 = np.dstack((err_l[0.5]['aupr'], err_s[0.5]['aupr']))
    o1 = np.dstack((err_l[1]['aupr'], err_s[1]['aupr']))

    xax = pd.Series(lamSs, name="lamS")
    conds = pd.Series(["Fused L2", "Fused SCAD"], name="method")

    plt.subplot(131)
    sns.tsplot(o0, time=xax, condition=conds, value="AUPR")
    plt.axis([0,4,0.8,0.91])
    plt.title("Orth error = 0")

    plt.subplot(132)
    sns.tsplot(o05, time=xax, condition=conds, value="AUPR")
    plt.axis([0,4,0.8,0.91])
    plt.title("Orth error = 0.5")

    plt.subplot(133)
    sns.tsplot(o1, time=xax, condition=conds, value="AUPR")
    plt.axis([0,4,0.8,0.91])
    plt.title("Orth error = 1")

    plt.show()

    return (err_l, err_s)

def test_scad_nonopt_params():
#create simulated data set with false orthology and run fused L2 and fused scad; faster than test_scad_opt_params because doesn't search through a
    import copy    

    N_TF = 20
    N_G = 200
    amt_fused = 0.5
    orth_err = [0,0.5,0.75,1]
    lamSs = [0,2,4]#[0, 2, 4, 6]
    seed = 10
    reps = 1
    k = 5

    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con','B_mse']
    err_dict_s = {m : np.zeros((k, len(lamSs))) for m in metrics}
    err_dict_l = {m : np.zeros((k, len(lamSs))) for m in metrics}
    err_s = {o : copy.deepcopy(err_dict_s) for o in orth_err}
    err_l = {o : copy.deepcopy(err_dict_l) for o in orth_err}

    if not os.path.exists(os.path.join('data','fake_data','test_scad_nonopt_params')):
        os.mkdir(os.path.join('data','fake_data','test_scad_nonopt_params'))
    #iterate over how much fusion

    for p in range(reps):
        for i, N in enumerate(orth_err):
            out = os.path.join('data','fake_data','test_scad_nonopt_params','dat_'+str(N))
            ds.write_fake_data1(N1 = 5*5, N2 = 5*50, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1)
            lamR = 2
            lamP = 1.0 #priors don't matter

            for j, lamS in enumerate(lamSs):
                print N
                print lamS
                (errd1l, errd2l) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=5, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
                print errd1l['aupr']
                for metric in metrics:
                    err_l[N][metric][:, [j]] = errd1l[metric]
                print err_l[N]['aupr']

                scad_settings = fr.get_settings({'s_it':50, 'a':1.0})
                (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=5, solver='solve_ortho_direct_scad', reverse = False, settings = scad_settings, cv_both = (True, True))
                print errd1['aupr']
                for metric in metrics:
                    err_s[N][metric][:, [j]] = errd1[metric]
                print err_s[N]['aupr']

    o0 = np.dstack((err_l[0]['aupr'], err_s[0]['aupr']))
    o05 = np.dstack((err_l[0.5]['aupr'], err_s[0.5]['aupr']))
    o1 = np.dstack((err_l[1]['aupr'], err_s[1]['aupr']))

    xax = pd.Series(lamSs, name="lamS")
    conds = pd.Series(["Fused L2", "Fused SCAD"], name="method")

    plt.subplot(131)
    sns.tsplot(o0, time=xax, condition=conds, value="AUPR")
    plt.title("Orth error = 0")

    plt.subplot(132)
    sns.tsplot(o05, time=xax, condition=conds, value="AUPR")
    plt.title("Orth error = 0.5")

    plt.subplot(133)
    sns.tsplot(o1, time=xax, condition=conds, value="AUPR")
    plt.title("Orth error = 1")

    return (err_l, err_s)


def test_scad_quick():
#create simulated data set with false orthology and run fused L2 and fused scad; faster than test_scad_opt_params2 because doesn't search through a
     
    N_TF = 20
    N_G = 200#50
    #N_TF = 10
    #N_G = 30
    amt_fused = 0.5
    orth_err = [0,0.75,1.0]#,0.25,0.5,0.75,1]
    lamSs = [0,3,6]#[0, 2, 4, 6]
    seed = 10
    reps = 10
    N_conds = 30
    if not os.path.exists(os.path.join('data','fake_data','test_scad_quick')):
        os.mkdir(os.path.join('data','fake_data','test_scad_quick'))
    #iterate over how much fusion
    errors_scad = np.zeros((len(orth_err), len(lamSs)))
    errors_l2 = np.zeros((len(orth_err), len(lamSs)))
    errors_scad_var = np.zeros((len(orth_err), len(lamSs)))
    errors_l2_var = np.zeros((len(orth_err), len(lamSs)))
    
    for p in range(reps):
        for i, N in enumerate(orth_err):
            out = os.path.join('data','fake_data','test_scad_quick','dat_'+str(N))
            ds.write_fake_data1(N1 = 20*10, N2=50*10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.0)
            lamR = 2
            lamP = 1.0 #priors don't matter

            for j, lamS in enumerate(lamSs):
                (errd1l, errd2l) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = False, seed=seed, cv_both = (True, True))
                #print lamS
                #print errd1l
                errors_l2[i,j] += errd1l['aupr'].mean()
                errors_l2_var[i,j] += errd1l['aupr'].mean()**2                
                scad_settings = fr.get_settings({'s_it':50, 'per':((1/(1+float(N)))*100)})
                (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_scad', reverse = False, seed=seed, settings=scad_settings, cv_both = (True, True))
                #print errd1
                errors_scad[i,j] += errd1['aupr'].mean()
                errors_scad_var[i,j] += errd1['aupr'].mean()**2

    errors_scad /= reps
    errors_l2 /= reps
    errors_scad_var /= reps
    errors_scad_var -= errors_scad**2
    errors_l2_var /= reps
    errors_l2_var -= errors_l2**2

    #return (errors_l2,errors_l2_var,errors_scad,errors_scad_var)

    #(errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=2, solver='solve_ortho_direct', reverse = False, seed=seed, cv_both = (True, True))
    settings = fr.get_settings({'s_it':2, 'per':((1/(1+float(N)))*100), 'return_cons':True})
    (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=2, solver='solve_ortho_direct_scad', reverse = False, seed=seed, settings = settings, cv_both = (True, True))
    
    errors_l2[i,j] += errd1['B_mse'].mean()
    plot_fuse_lams(out, settings['cons'])
    #return (errors_l2)
    #return (errors_scad, errors_scad_var, errors_l2, errors_l2_var)

#visualizes lams at each oon of scad
def plot_scad_iteration():
    N_TF = 10
    N_G = 50
    amt_fused = 0.5
    orth_err = [1]
    lamSs = [3]
    reps = 1

    if not os.path.exists(os.path.join('data','fake_data','plot_scad_iteration')):
        os.mkdir(os.path.join('data','fake_data','plot_scad_iteration'))

    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','plot_scad_iteration','dat_'+str(N))
        ds.write_fake_data1(N1 = 4*5, N2 = 4*50, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1)
        lamR = 2
        lamP = 1.0 #priors don't matter

        for j, lamS in enumerate(lamSs):
            scad_settings = fr.get_settings({'s_it':20, 'per':((1/(1+float(N)))*100)})
            print N
            print lamS
            (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=4, solver='solve_ortho_direct_scad_plot', reverse = False, settings = scad_settings, cv_both = (True, True))
            print errd1

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
    ds.write_fake_data1(N1 = 10, N2 = 10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = orth_err, orth_falseneg = orth_err, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
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
        lamSs = []

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
        fuse_constraints = fr.orth_to_constraints(organisms, genes, tfs, orth, 0.5)
        #print fuse_constraints
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
            print B1u-B2u
            deltabeta.append(B1u-B2u)
            s = pair.lam
            print s
            fusionpenalty.append(s*((B1u-B2u)**2))
            lamSs.append(s)

        plt.scatter(deltabeta, fusionpenalty)
        plt.xlabel('delta beta')
        plt.ylabel('penalty')
        plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad2',str(fold))))
        plt.figure()
        plt.clf()
        plt.clf()
        plt.hist(lamSs, 50)
        plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad2',str(fold)+'lamS')))

def test_mcp3():
#create simulated data set with false orthology and run fused L2 and fused scad; based off of test_scad()
     
    N_TF = 10
    N_G = 200
    amt_fused = 0.5
    orth_err = [0.25,0.5,0.75,1]
    lamSs = [0, 2, 4]
    if not os.path.exists(os.path.join('data','fake_data','test_mcp3')):
        os.mkdir(os.path.join('data','fake_data','test_mcp3'))
    #iterate over how much fusion
    errors_mcp = np.zeros((len(orth_err), len(lamSs)))
    errors_l2 = np.zeros((len(orth_err), len(lamSs)))
    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','test_mcp3','dat_'+str(N))
        #ds.write_fake_data1(N1 = 20, N2 = 20, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
        ds.write_fake_data1(N1 = 5, N2 = 5, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1)
        #lamR = 4
        lamR = 1
        lamP = 1.0 #priors don't matter
        
        for j, lamS in enumerate(lamSs):
            mcp_settings = fr.get_settings({'m_it':50, 'a':0.6})

            (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=2, solver='solve_ortho_direct_mcp', reverse = False, settings = mcp_settings, cv_both = (True, True))
            (errd1l, errd2l) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=2, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
            errors_mcp[i,j] = errd1['aupr'].mean()
            errors_l2[i,j] = errd1l['aupr'].mean()

    return (errors_mcp, errors_l2)

def test_mcp():
#create simulated data set with false orthology and run fused scad + visualize mcp penalty at each cv
     
    N_TF = 10
    N_G = 200
    amt_fused = 0.5
    orth_err = 1.0
    lamS = 2

    if not os.path.exists(os.path.join('data','fake_data','test_mcp')):
        os.mkdir(os.path.join('data','fake_data','test_mcp'))

    out = os.path.join('data','fake_data','test_mcp','ortherr'+str(orth_err))
    ds.write_fake_data1(N1 = 2, N2 = 20, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = orth_err, orth_falseneg = orth_err, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
    lamR = 5
    lamP = 1.0 #priors don't matter

    k = 2
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
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=N, k=10, solver='solve_ortho_direct', reverse = True, cv_both = (True, False))
        errors[i] = errd1['aupr'].mean()
    
    plt.plot(lamSs, errors, 'ro')
    plt.savefig(os.path.join(os.path.join('data','bacteria_standard','studentseminar','fig1')))
    plt.figure()

def orth_err_perf():
    N_TF = 20
    N_G = 200
    amt_fused = 0.5
    orth_err = [1]
    lamSs = [0,2,4]#[0, 2, 4, 6]
    seed = 10
    if not os.path.exists(os.path.join('data','fake_data','ortherrperf')):
        os.mkdir(os.path.join('data','fake_data','ortherrperf'))

    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','ortherrperf','dat_'+str(N))
        ds.write_fake_data1(N1 = 5*5, N2 = 5*50, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1)
        lamR = 2
        lamP = 1.0 #priors don't matter

        for j, lamS in enumerate(lamSs):
            print N
            print lamS
            (errd1l, errd2l) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=5, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
            print errd1l['aupr'].mean()


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
    new_cons = fr.scad([b1, b2], constraints, lamS, a=a)
    new_con_val = np.array(map(lambda x: x.lam, new_cons))
    plt.plot(b1, new_con_val)
    plt.show()

#plots constraints of simulated data after scad adjustment
def test_scad1b(lamS=2,a=0.3):
    out = os.path.join('data','fake_data','test_scad','dat_0.5')
    (constraints, marks, orth) = ds.load_constraints(out)
    (b1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (b2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
#    con_val = np.array(map(lambda x: x.lam, constraints))
    new_cons = fr.scad([b1, b2], constraints, lamS, a=a)
    new_con_val = np.array(map(lambda x: x.lam, new_cons))
    deltabeta = []
    for con in new_cons:
        b1_val = b1[con.c1.r, con.c1.c]
        b2_val = b2[con.c2.r, con.c2.c]
        deltabeta.append(np.abs(b1_val-b2_val))
    deltabeta_ar = np.array(deltabeta)
    plt.scatter(deltabeta_ar, new_con_val)
    plt.show()
#    plt.clf()
#    plt.scatter(deltabeta_ar, con_val)
#    plt.show()

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
        
        settings = fr.get_settings({'s_it':5, 'return_cons':True, 'a':0.1})
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS, solver='solve_ortho_direct_scad',settings = settings)
        plt.plot(b1[0,2], b1[1,2], 'or',markersize=0.5*(10+20*lamS))
        plt.plot(b2[0,2], b2[1,2], 'ob',markersize=0.5*(10+20*lamS))
        
        assemble_orths(settings['cons'])
        
        
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
        
        settings = fr.get_settings({'s_it':5, 'return_cons':True, 'a':1.50})
        (b1, b2) = fg.fit_model(out, lamPs[0], lamRs[0], lamS, solver='solve_ortho_direct_scad',settings = settings)
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
    ds.write_fake_data1(N1 = N1, N2 = N2, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.0)
    def waka():        
        (b1, b2) = fg.fit_model(out, 1.0, 0.1, 0.5, solver='solve_ortho_direct')
    def waka2():
        plot_bacteria_roc(lamP=1.0, lamR=5, lamSs=[0,10], k=1, metric='roc',savef='profiletest',normed=False,scad=False, cv_both=(True,False,True), roc_species=0, orgs=None, unfused=False, lamS_opt=None, orth_file=['orth','operon'])
    import profile
    profile.runctx("waka2()",globals(),locals(),sort='cumulative',filename=os.path.join(out,'stats'))
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
    seed = 5
    out = os.path.join('data','fake_data','struct1')
    ds.write_fake_data1(N1 = N1, N2 = N2, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=sparse, fuse_std = 0.0)
    
    lamP = 1.0
    #lamR = 100
    #lamS = 0.0001
    lamR = 4
    lamS = 2
    k=5
    cv_both = (True, True)
    (errd11, errd12) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = False, cv_both = (True, True), pct_priors=0.75, seed=seed)
    lamP = 0.001
    (errd21, errd22) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = False, cv_both = (True, True), pct_priors=0.75, seed=seed)
    
    print errd11['aupr']
    print errd21['aupr']
    #(b1, b2) = fg.fit_model(out, lamP, lamR, lamS, solver='solve_ortho_direct')
    #(b1r, g, t) = ds.load_network(os.path.join(out, 'beta1'))
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
                (errd1, errd2) = fg.cv_model_m(out2, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
                errors[i, j] += errd1['R2'].mean()
    
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


def increase_data3():
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
                (errd1, errd2) = fg.cv_model_m(out2, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
                errors[i, j] += errd1['aupr'].mean()
    
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


def plot_betas_scad(net_path):
    lamR = 2
    lamP = 1
    lamS = 4
    out2 = os.path.join('data','fake_data','plot_betas_scad')
    if not os.path.exists(out2):
        os.mkdir(out2)
    #out = os.path.join('data','fake_data','plot_betas2','dat')
    ds1 = ds.standard_source(net_path,0)
    ds2 = ds.standard_source(net_path,1)
    orth_fn = os.path.join(net_path, 'orth')
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

    settings_fs=fr.get_settings({'s_it': 30, 'per':((1/(1+float(0.5)))*100), 'return_cons':True})
    Bs_fs = fr.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, settings = settings_fs)

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
    Bufd = []
    Bfrd = []
    Bfsd = []

    colors = []
    con_inds = np.random.permutation(range(len(constraints)))
    area_fr = []
    area_fs = []
    area_fm = []
    cons = []
    cons_fs = []

    subs = 2000
    for i in con_inds:#[0:subs]:
        con = constraints[i]
        cons.append(con.lam)
        mark = marks[i]
        Buf1.append(Bs_uf[con.c1.sub][con.c1.r, con.c1.c])
        Buf2.append(Bs_uf[con.c2.sub][con.c2.r, con.c2.c])
        Bufd.append(Bs_uf[con.c1.sub][con.c1.r, con.c1.c]-Bs_uf[con.c2.sub][con.c2.r, con.c2.c])
        Bfr1.append(Bs_fr[con.c1.sub][con.c1.r, con.c1.c])
        Bfr2.append(Bs_fr[con.c2.sub][con.c2.r, con.c2.c])
        Bfrd.append(Bs_fr[con.c1.sub][con.c1.r, con.c1.c]-Bs_fr[con.c2.sub][con.c2.r, con.c2.c])

        con_fs = settings_fs['cons'][i]
        cons_fs.append(con_fs.lam)
        Bfs1.append(Bs_fs[con_fs.c1.sub][con_fs.c1.r, con_fs.c1.c])
        Bfs2.append(Bs_fs[con_fs.c2.sub][con_fs.c2.r, con_fs.c2.c])
        Bfsd.append(Bs_fs[con_fs.c1.sub][con_fs.c1.r, con_fs.c1.c]-Bs_fs[con.c2.sub][con.c2.r, con.c2.c])

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
        colors.append('b')


    Buf1s = np.array(Buf1)
    Buf2s = np.array(Buf2)
    Bufds = np.array(Bufd)
    Bfr1s = np.array(Bfr1)
    Bfr2s = np.array(Bfr2)
    Bfrds = np.array(Bfrd)
    Bfs1s = np.array(Bfs1)
    Bfs2s = np.array(Bfs2)
    Bfsds = np.array(Bfsd)

    plt.close()
    plt.hist(cons_fs)
    plt.title('scad lamS dist')    
    plt.savefig(os.path.join(out2,'scad lamS dist'))
    plt.show(block=False)
#    return (Bfsds, cons_fs)    
#    print cons_fs
#    with sns.axes_style("white"):
#        sns.jointplot(Bfsds, np.array(cons_fs), kind = "hex");
#    plt.show()

    plt.figure()
    plt.scatter(Buf1s, Buf2s, c=colors, alpha=0.5)
    plt.xlabel('beta network 1')
    plt.ylabel('beta network 2')
    plt.title('unfused')
    plt.savefig(os.path.join(out2,'unfused'))
    

    plt.figure()
    plt.scatter(Bfr1s, Bfr2s, c=colors, alpha=0.5)
    plt.xlabel('beta network 1')
    plt.ylabel('beta network 2')
    plt.title('fused l2')
    plt.savefig(os.path.join(out2,'fused l2'))
    
    plt.figure()
    plt.scatter(Bfs1s, Bfs2s, c=colors, alpha=0.5)
    plt.xlabel('beta network 1')
    plt.ylabel('beta network 2')
    plt.title('scad')
    plt.savefig(os.path.join(out2,'scad'))
   
    plt.show(block=False)

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
    ds.write_fake_data1(N1 = 15, N2 = 15, out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, orth_falsepos=0.7, orth_falseneg=0.7, fuse_std = 0.1, pct_fused=0.75)
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

    settings_fs = fr.get_settings({'a':1.5, 'return_cons':True})

    Bs_fs = fr.solve_ortho_direct_scad(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamSe, s_it, settings = settings_fs)
    m_it = 5

    settings_mcp = fr.get_settings({'return_cons':True})

    Bs_fm = fr.solve_ortho_direct_mcp(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamSe, m_it, settings = settings_mcp)
    em_it=10

    settings_em = fr.get_settings({'f':1,'uf':0.1})
    
    Bs_em = fr.solve_ortho_direct_em(organisms, genes, tfs, Xs, Ys, orth, priors, lamP, lamR, lamS, em_it, settings = settings_em)
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
        con = settings_fs['cons'][i]
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
        con = settings_fs['cons'][i]
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
        settings_em = fr.get_settings({'em_it':5, 'f':1, 'uf':1})
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct_em', reverse = True, settings = settings_em, cv_both = (True, True))

        
        (errl1, errl2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))

        (erru1, erru2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=0, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
        errors_em = np.hstack((errors_em, errd1['aupr']))
        errors_l2 = np.hstack((errors_l2, errl['aupr']))
        errors_uf = np.hstack((errors_uf, erru['aupr']))
    
    errors = np.dstack((errors_em, errors_l2, errors_uf))
    
    step = pd.Series(orth_err)
    solver = pd.Series(["EM", "fused L2", "L2"], name="solver")
    plt.close()
    sns.tsplot(errors, time=step, condition=solver, value="mean squared error")
    plt.savefig(os.path.join(os.path.join('data','fake_data','test_em1','fig5')))
    plt.show()
    return errors




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


    print 'there are %d mapped priors' % (len(priors_a) + len(priors_r))

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
    reps = 50

    lamRs = list(np.arange(0,5.0,0.5))
    lamSs = list(np.arange(0,10.0,1.0)) 
    lamP = 1.0
    mse_array = np.zeros((len(pct_fused),len(lamSs)))
    mse_var = np.zeros((len(pct_fused),len(lamSs)))
    best_Rs = np.zeros((len(pct_fused),len(lamSs)))

    out1 = os.path.join('data','fake_data','L2fusiontest')
    k = 2#cv folds
    if not os.path.exists(out1):
        os.mkdir(out1)

    for p in range(reps):
        for i, N in enumerate(pct_fused):
            out2 = os.path.join(out1,'dat_'+str(N)+str(p))
            ds.write_fake_data1(N1=10, N2=10, out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1, pct_fused=N)        
            lamP = 1.0 #priors don't matter
            for j, lamS in enumerate(lamSs):
                (mse, best_lamP, best_lamR, best_lamS, grid) = fg.grid_search_params(out2, lamP=[lamP], lamR=lamRs, lamS=[lamS], k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, True))
                mse_array[i,j] += mse
                mse_var[i,j] += mse**2
                best_Rs[i,j] = best_lamR
    mse_array /= k*p
    mse_var /= k*p
    mse_var -= mse_array**2
    return (mse_array, mse_var, best_Rs)
    mse_df = pd.DataFrame(mse_array, index=pct_fused, columns=lamSs)
    sns.heatmap(mse_df)
    plt.show()

#makes heatmap showing effect of lamS on networks in L2 fusion; does not use optimal lamR
def L2fusion_quick():    
    N_TF = 20
    N_G = 200
    pct_fused = list(np.linspace(0.3,1.0,10))
    reps = 2
    
    lamR = 2
    lamSs = list(np.linspace(0,3,10)) 
    lamP = 1.0
    aupr_array = np.zeros((len(pct_fused),len(lamSs)))

    out1 = os.path.join('data','fake_data','L2fusion_quick')
    k = 2#cv folds
    if not os.path.exists(out1):
        os.mkdir(out1)

    for p in range(reps):
        for i, N in enumerate(pct_fused):
            out2 = os.path.join(out1,'dat_'+str(N))
            ds.write_fake_data1(N1=2*8, N2=50*8, out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.75, fuse_std = 0.35, pct_fused=N, orth_falsepos = 0)        
            lamP = 1.0 #priors don't matter
            seed = 10
            for j, lamS in enumerate(lamSs):
                (errd1, errd2) = fg.cv_model_m(out2, lamP, lamR, lamS, k, solver='solve_ortho_direct', reverse=False, cv_both=(True,True), exclude_tfs=True, pct_priors=0, seed=seed, verbose=False)
                #print errd1['corr'].mean()
                print errd1['R2'].mean()
                aupr_array[i,j]+= errd1['R2'].mean()

    aupr_array /= reps
    ar2 = np.zeros((aupr_array.shape[0],aupr_array.shape[1]))
    for i in range(len(aupr_array)):
        ar2[i,:] = aupr_array[i,:]/aupr_array[i,0]
        #ar2[i,:] = aupr_array[i,:]
    df = pd.DataFrame(ar2[1:,:],index=pct_fused[1:],columns=lamSs)
    df.index.name = 'percent fused'
    df.columns.name = 'lamS'
    sns.heatmap(df,cmap="Blues", square=True)
    plt.show()

def L2plot():
    N_TF = 20
    N_G = 200
    pct_fused = list(np.linspace(0,1.0,20))

    lamR = 2
    lamSs = [0,list(np.linspace(0,3,10))[4]]
    lamP = 1.0
    aupr_array = np.zeros((len(pct_fused),len(lamSs)))
    aupr_adj = np.zeros((len(pct_fused),len(lamSs)))

    reps = 20

    out1 = os.path.join('data','fake_data','l2plot')
    k = 10#cv folds
    if not os.path.exists(out1):
        os.mkdir(out1)

    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dicts = [{m : np.zeros((k*reps, len(pct_fused))) for m in metrics},{m : np.zeros((k*reps, len(pct_fused))) for m in metrics}]
    err1_adj = np.zeros((k*reps,len(pct_fused)))
    err2_adj = np.zeros((k*reps,len(pct_fused)))

    for p in range(reps):
        for i, N in enumerate(pct_fused):
            print N
            out2 = os.path.join(out1,'dat_'+str(N))
            ds.write_fake_data1(N1=5*k, N2=50*k, out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.75, fuse_std = 0.1, pct_fused=N)        
            lamP = 1.0 #priors don't matter
            seed = 10
            for j, lamS in enumerate(lamSs):
                print lamS
                (errd1, errd2) = fg.cv_model_m(out2, lamP, lamR, lamS, k, solver='solve_ortho_direct',reverse=False, cv_both=(True,True), exclude_tfs=True, pct_priors=0, seed=seed, verbose=False)
                for metric in metrics:
                    err_dicts[j][metric][p*k:(p+1)*k, [i]] = errd1[metric]

    for j in range(err_dicts[0]['aupr'].shape[0]):
        for k in range(err_dicts[0]['aupr'].shape[1]):
            err1_adj[j,k] = err_dicts[0]['aupr'][j,k]/err_dicts[0]['aupr'][j,:].mean()

    for j in range(err_dicts[1]['aupr'].shape[0]):
        for k in range(err_dicts[1]['aupr'].shape[1]):
            err2_adj[j,k] = err_dicts[1]['aupr'][j,k]/err_dicts[1]['aupr'][j,:].mean()

    to_plot = np.dstack((err1_adj, err2_adj))
    linedesc = pd.Series(['unfused','fused'],name='penalty')
    xs = pd.Series(pct_fused, name='network similarity')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='% aupr improvement')
    
    plt.show()
           
    return err_dicts
   


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
    
    plt.xlabel('beta x 1000')
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
    
    
    constraints_touch = set()
    
    (constraints, marks, orths) = ds.load_constraints(bactf)
    
    for con in constraints:
        constraints_touch.add(con.c1)
        constraints_touch.add(con.c2)

    p1c = fr.priors_to_constraints([ds1.name],[ds1.genes],[ds1.tfs],priors1,0.5)
    mappable_priors = filter(lambda x: x.c1 in constraints_touch, p1c)

    (e1_tr, t1_tr, genes1, tfs1) = ds1.load_data()
    (e2_tr, t2_tr, genes2, tfs2) = ds2.load_data()

    subt_to_anth = {orths[x].genes[0].name : orths[x].genes[1].name for x in range(len(orths))}
    anthracis_gene_inds = {genes2[i] : i for i in range(len(genes2))}
    subtilis_gene_inds = {genes1[i] : i for i in range(len(genes1))}
    print 'there are %d priors, of which %d map' % (len(priors1), len(mappable_priors))
    print 'there are %d orthology mappings, and %d constraints' % (len(orths), len(constraints)/2)
    print '%f percent of the subtilis network is constrained' % ((len(constraints)/2.0)/(len(genes1)*len(tfs1)))
    print '%f percent of the anthracis network is constrained' % ((len(constraints)/2.0)/(len(genes2)*len(tfs2)))
    
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
    #print 'there are %d different entries' % (B0_mapped != B0_mapped2).sum()
    
    aupr = fg.eval_network_pr(B0_mapped, genes1, tfs1, priors1, exclude_tfs=False)
    auc = fg.eval_network_roc(B0_mapped, genes1, tfs1, priors1, exclude_tfs=False)

    print 'performance, mapped: aupr %f, auc %f' % (aupr, auc)
    

    aupr = fg.eval_network_pr(B0_mapped2, genes1, tfs1, priors1, exclude_tfs=False, constraints=constraints, sub=0)
    auc = fg.eval_network_roc(B0_mapped2, genes1, tfs1, priors1, exclude_tfs=False, constraints=constraints, sub=0)

    print 'performance-constr, mapped2: aupr %f, auc %f' % (aupr, auc)

#plots the chosen metric as a function of lamS for bacterial data    
#DEPRACATED: use plot_bacteria, which plots from saved error dictionaries
def plot_bacteria_performance(lamP=1.0, lamR=5, lamSs=[0,1,2,3,4], k=20):

    out = 'data/bacteria_standard'
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamSs))) for m in metrics} #anthracis

    rseed = random.random()
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=0.0, verbose=True, seed=rseed)

        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]

    #to_plot = err_dict1['aupr']#
    metric = 'auc'
    to_plot = np.dstack((err_dict1[metric], err_dict1[metric+'_con']))


    linedesc = pd.Series(['full','constrained'],name='error type')
    #linedesc = pd.Series(['full'],name='error type')
    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value=metric)
    with file('err_dict1','w') as f:
        pickle.dump(err_dict1, f)
    
    plt.show()


#plots performance as a function of R
def plot_bacteria_performanceR(lamP=1.0, lamRs=[1,4,7,10,13], lamS=0, k=20):

    out = 'data/bacteria_standard'
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamRs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamRs))) for m in metrics} #anthracis

    rseed = random.random()
    for i, lamR in enumerate(lamRs):
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=0.0, verbose=True, seed=rseed)

        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]

    #auprs = err_dict1['aupr'].mean(axis=0)
    auprs = err_dict1['aupr'].mean(axis=0)
    #stes = err_dict1['aupr'].std(axis=0) / err_dict1['aupr'].shape[0]**0.5
    stes = err_dict1['aupr'].std(axis=0) / err_dict1['aupr'].shape[0]**0.5
    pretty_plot_err(lamRs, auprs, stes, (1, 0.5, 0, 0.25))

def plot_bacteria_performance_opt_param():

    lamP = 1.0
    lamRs = list(np.arange(0,10.0,1.0))
    lamSs = list(np.arange(0,10.0,1.0))
    k=10

    out = 'data/bacteria_standard'
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    aupr_array = np.zeros((len(lamSs),1))
    lamR_array = np.zeros((len(lamSs),1))

    rseed = random.random()
    for i, lamS in enumerate(lamSs):
        (aupr, best_lamP, best_lamR, best_lamS, grid) = fg.grid_search_params(out, [lamP], lamRs, [lamS], k, solver='solve_ortho_direct', reverse=False, cv_both=(True,False), exclude_tfs=False, eval_metric='aupr')
        aupr_array[i,0] = aupr
        lamR_array[i,0] = best_lamR
    
    return (aupr_array, lamR_array)

    #to_plot = err_dict1['aupr']#
    to_plot = np.dstack((err_dict1['aupr'], err_dict1['aupr_con']))


    linedesc = pd.Series(['full','constrained'],name='error type')
    #linedesc = pd.Series(['full'],name='error type')

    xs = pd.Series(lamRs, name='lamR')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='aupr')
    #with file('err_dict1','w') as f:
    #    pickle.dump(err_dict1, f)
    
    plt.show()



    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='aupr')
    #with file('err_dict1','w') as f:
    #    pickle.dump(err_dict1, f)
    plt.savefig(os.path.join('data','bacteria_standard','opt_params'))
    plt.show()


#functions as plot_bacteria_performance, plotting aupr (on constrained interactions) as a function of lamS, but does so with half the priors used in training, and compares lamP= supplied lamP to lamP=1
def plot_bacteria_performance_priors(lamP=1.0, lamR=5, lamSs=[0,2,4,6], k=20):

    out = 'data/bacteria_standard'
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']

    no_pr = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    yes_pr = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    pct_priors = 0.5
    rseed = random.random()
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model_m(out, lamP=1.0, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=pct_priors, verbose=True, seed=rseed)
        for metric in metrics:
            no_pr[metric][:, [i]] = errd1[metric]

        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=pct_priors, verbose=True, seed=rseed)
        for metric in metrics:
            yes_pr[metric][:, [i]] = errd1[metric]

    to_plot = np.dstack((no_pr['aupr_con'], yes_pr['aupr_con']))


    linedesc = pd.Series(['no priors','priors'],name='error type')
    
    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='aupr')
    #with file('err_dict1','w') as f:
    #    pickle.dump(err_dict1, f)
    
    plt.show()

#doesn't use fusion and varies the amount of priors used in training
def plot_bacteria_performance_priors2():
    lamP1 = 0.3
    lamP2 = 0.1
    lamP3 = 0.05
    lamR = 10
    lamS = 0
    k = 10
    pct_priors = np.arange(0.2,0.8,0.2)

    out = 'data/bacteria_standard'
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']

    pr1 = {m : np.zeros((k, len(pct_priors))) for m in metrics}
    pr2 = {m : np.zeros((k, len(pct_priors))) for m in metrics}
    pr3 = {m : np.zeros((k, len(pct_priors))) for m in metrics}
    no_pr = {m : np.zeros((k, len(pct_priors))) for m in metrics} 
    rseed = random.random()


    for i, priors in enumerate(pct_priors):
        (errd1, errd2) = fg.cv_model_m(out, lamP=1.0, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=priors, verbose=True, seed=rseed)
        for metric in metrics:
            no_pr[metric][:, [i]] = errd1[metric]

        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP1, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=priors, verbose=True, seed=rseed)
        for metric in metrics:
            pr1[metric][:, [i]] = errd1[metric]

        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP2, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=priors, verbose=True, seed=rseed)
        for metric in metrics:
            pr2[metric][:, [i]] = errd1[metric]


        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP3, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=priors, verbose=True, seed=rseed)
        for metric in metrics:
            pr3[metric][:, [i]] = errd1[metric]


    to_plot = np.dstack((no_pr['R2'], pr1['R2'], pr2['R2'], pr3['R2']))


    linedesc = pd.Series(['lamP = 1', 'lamP = 0.3', 'lamP = 0.1', 'lamP = 0.05'],name='lamP')
    
    xs = pd.Series(pct_priors, name='pct priors')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='R2')
    #with file('err_dict1','w') as f:
    #    pickle.dump(err_dict1, f)
    
    plt.show()

#functions as plot_bacteria_performance_priors, plotting aupr (on constrained interactions) as a function of lamS, but does so with half the priors used in training, and compares lamP= supplied lamP to lamP=1; uses synthetic data
def plot_synthetic_performance_priors(lamP=1.0, lamR=5, lamSs=[0,2,4,6,8,10,12], k=2):
    N = 8
    N_TF = 15
    N_G = 300
    out = os.path.join('data','fake_data','plot_synthetic_performance_priors')
    ds.write_fake_data1(N1 = N, N2 = N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.8, sparse=0.9, fuse_std = 0.0, orth_falsepos=0.0,orth_falseneg=0.0)

    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']

    no_pr = {m : np.zeros((k, len(lamSs))) for m in metrics}
    yes_pr = {m : np.zeros((k, len(lamSs))) for m in metrics}
    pct_priors = 0.5
    rseed = random.random()

    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model_m(out, lamP=1.0, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,True), exclude_tfs=True, pct_priors=pct_priors, verbose=True, seed=rseed)
        for metric in metrics:
            no_pr[metric][:, [i]] = errd1[metric]

        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,True), exclude_tfs=True, pct_priors=pct_priors, verbose=True, seed=rseed)
        for metric in metrics:
            yes_pr[metric][:, [i]] = errd1[metric]

    to_plot = np.dstack((no_pr['mse'], yes_pr['mse']))

    #linedesc = pd.Series(['no priors','priors'],name='error type')
    
    #xs = pd.Series(lamSs, name='lamS')
    #sns.tsplot(to_plot, time=xs, condition=linedesc, value='aupr')
    #with file('err_dict1','w') as f:
    #    pickle.dump(err_dict1, f)
    plt.plot(lamSs, no_pr['aupr'].mean(axis=0), label='no prior')
    plt.plot(lamSs, yes_pr['aupr'].mean(axis=0), label='prior')
    plt.xlabel('lamS')
    plt.legend()
    plt.show()

#solves a model on its own, then simultaneously with no lamS
def solve_both_ways(out, lamP, lamR, lamS):
    d1 = ds.standard_source(out,0)
    d2 = ds.standard_source(out,1)
    (e1, t1, genes1, tfs1) = d1.load_data()
    (e2, t2, genes2, tfs2) = d2.load_data()
    (constraints, marks, orths) = ds.load_constraints(out)
    
    Bs0 = fr.solve_ortho_direct(['uno'],[genes1],[tfs1],[t1],[e1],[],[],lamP, lamR, lamS)
    Bs1 = fr.solve_ortho_direct(['uno','dos'],[genes1,genes2],[tfs1,tfs2],[t1,t2],[e1,e2],orths,[],lamP, lamR, lamS)
    #plt.matshow(e2)
    #plt.show()
    #plt.matshow(t2)
    #plt.show()
    return (Bs0[0], Bs1[0])
    
def plot_synthetic_beta_err(N = 10, lamP=1, lamR=5,lamSs=[0,1]):
    
    N_TF = 20
    N_G = 50
    out = os.path.join('data','fake_data','plot_synthetic_beta_err')

    ds.write_fake_data1(N1 = N, N2 = N*10, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, pct_fused=1.0, sparse=0.5, fuse_std = 0.0, orth_falsepos=0.0,orth_falseneg=0.0)
    
    net1 = ds.load_network(out+'/beta1')[0]
    net2 = ds.load_network(out+'/beta2')[0]
    k=2
    net_errs = []
    auprs = []
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False)
        auprs.append(errd1['aupr'].mean())

        (est1, est2) = fg.fit_model(out, lamP=lamP, lamR=lamR, lamS=lamS)
        
        (a, b) = solve_both_ways(out, lamP, lamR, lamS)
        print est1
        print net1
        print 'wtf is going on'
        print ((a-b)**2).mean()
        print ((a-est1)**2).mean()
        print ((est1-net1)**2).mean()
        print ((est2-net2)**2).mean()
        #plt.matshow(est2)
        #plt.colorbar()
        #plt.show()
        err = 0
        err += ( (est1 - net1)**2 ).mean()
        err += ( (est2 - net2)**2 ).mean()
        net_errs.append(err)
    plt.plot(lamSs, net_errs)
    plt.show()
    plt.plot(lamSs, auprs)
    plt.show()

#plots error dictionaries
#this is really for debugging plot_bacteria_performance 
def plot_synthetic_performance(lamP=1.0, lamR=5, lamSs=[0,1], k=20, fuse_std=0.0):
    N = 10
    N_TF = 20
    N_G = 30
    out = os.path.join('data','fake_data','plot_synthetic_performance')
    ds.write_fake_data1(N1 = k*N, N2 = 5*k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.8, sparse=0.5, fuse_std = fuse_std, orth_falsepos=0.0,orth_falseneg=0.0)
    
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamSs))) for m in metrics} #anthracis
    
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False)
        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]

    measures = ('auc','auc_con')
    to_plot = np.dstack((err_dict1[measures[0]], err_dict1[measures[1]]))
    linedesc = pd.Series(measures,name='error type')
    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value=measures[0])
    
    plt.show()

#given a savef, plots whatever metric you like as a function of lamR, lamP, or lamS
#lamS here temporarily
def plot_savef(savef, metric, lamSs):
    errdls = []
    scores = []
    import pickle
    savef = os.path.join('saves',savef)
    with file(savef) as f:
        errdls = pickle.load(f)
    
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = errdls[i]        
        scores.append(errd1[metric].mean())
    
    plt.plot(lamSs, scores)
    plt.show()

    return None
    
    

#as plot synthetic performance, except it plots overlaid ROC curves for different values of lamS
def plot_synthetic_roc(lamP=1.0, lamR=5, lamSs=[0,1], k=20, metric='roc',normed=False):
    N = 10
    N_TF = 20
    N_G = 30
    out = os.path.join('data','fake_data','plot_synthetic_performance')
    ds.write_fake_data1(N1 = k*N, N2 = 5*k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.8, sparse=0.5, fuse_std = 0.0, orth_falsepos=0.0,orth_falseneg=0.0)
    plot_roc(out, lamP, lamR, lamSs, k, metric,normed=normed)

#as plot synthetic performance, except it plots overlaid ROC curves for different values of lamS
def plot_bacteria_roc(lamP=1.0, lamR=5, lamSs=[0,1], k=20, metric='roc',savef=None,normed=False,scad=False, cv_both=(True,False), roc_species=0, orgs=None, unfused=False, lamS_opt=None, orth_file=['orth']):
    out = os.path.join('data','bacteria_standard')
    plot_roc(out, lamP, lamR, lamSs, k, metric, savef,normed,scad, cv_both, roc_species, orgs, lamS_opt, unfused, orth_file)

#generic function for above two

#use a list of dictionaries for lamS_opt if using more than one lamS_opt; always include all pairwise lamS for fused species  
def plot_roc(out, lamP=1.0, lamR=5, lamSs=[0,1], k=20, metric='roc',savef=None,normed=False,scad=False, cv_both=(True,False), roc_species=0, orgs=None, lamS_opt = None, unfused = False, orth_file=['orth']):

    seed = np.random.randn()
    scad = False
    if scad:
        settings = fr.get_settings({'per' : 75})
        solver = 'solve_ortho_direct_scad'
    else:
        settings = fr.get_settings()
        solver = 'solve_ortho_direct'

    if metric[0:3] == 'roc':
        xl = 'false alarm'
        yl = 'hit'
        summary_measure = 'auc'
    if metric[0:3] == 'prc':
        xl = 'recall'
        yl = 'precision'
        summary_measure = 'aupr'
    
    if 'con' in metric:
        chancek = 'chance_con'
        summary_measure = summary_measure + '_con'
    else:
        chancek = 'chance'
        #summary_measure = summary_measure + '_con'
    all_roc_curves = []
    import pickle
    if savef != None:
        savef = os.path.join('saves',savef)
    loaded = False
    if savef != None and os.path.exists(savef):
        with file(savef) as f:
            loaded = True
            errdls = pickle.load(f)
    else:
        errdls = []

        if unfused == True:
            errd = fg.cv_unfused(out, lamP=lamP, lamR=lamR, k=k, solver = solver, settings = settings, reverse = True, cv_both = cv_both, exclude_tfs=False, seed = seed, orgs = orgs, lamS_opt = lamS_opt)
            errdls.append(errd)
            for i, lamS in enumerate(lamSs):
                errdf = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver = solver, settings = settings, reverse = True, cv_both = cv_both, exclude_tfs=False, seed = seed, orgs = orgs, lamS_opt = None, orth_file = orth_file)[roc_species]
                errdls.append(errdf)
            if lamS_opt != None:
                errdf = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver = solver, settings = settings, reverse = True, cv_both = cv_both, exclude_tfs=False, seed = seed, orgs = orgs, lamS_opt = lamS_opt, orth_file = orth_file)[roc_species]
                errdls.append(errdf)
        
        else:
            for i, lamS in enumerate(lamSs):
                print i
                errdf = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver = solver, settings = settings, reverse = True, cv_both = cv_both, exclude_tfs=False, seed = seed, orgs = orgs, lamS_opt = None, orth_file = orth_file)[roc_species]
                errdls.append(errdf)
            if lamS_opt != None:
                errdf = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver = solver, settings = settings, reverse = True, cv_both = cv_both, exclude_tfs=False, seed = seed, orgs = orgs, lamS_opt = lamS_opt, orth_file = orth_file)[roc_species]
                errdls.append(errdf)
                
    print len(errdls)
    if savef != None and not loaded:
        with file(savef, 'w') as f:
            pickle.dump(errdls, f)

    chance_rates = []

    if unfused == True:
        errd = errdls[0]
        rocs = errd[metric]
        print (errd[summary_measure].mean())
        all_roc_curves.append(rocs)
        chance_rates.append(errd[chancek])
        for i, lamS in enumerate(lamSs):
            errdf = errdls[i+1]      
            rocs = errdf[metric]
            print (lamS, errdf[summary_measure].mean())
            all_roc_curves.append(rocs)
            chance_rates.append(errdf[chancek])
        errd_opt = errdls[-1]
        rocs_opt = errd_opt[metric]
        print (lamS_opt, errd_opt[summary_measure].mean())
        all_roc_curves.append(rocs_opt)
        chance_rates.append(errd_opt[chancek])

        toplots = ['rank combined']
        toplots.extend(map(str, lamSs))
        toplots.append(str(lamS_opt))
        linedesc = pd.Series(toplots, name='method') 
    
    else:
        for i, lamS in enumerate(lamSs):
            print lamS
            print i
            errd = errdls[i]      
            rocs = errd[metric]
            print (lamS, errd[summary_measure].mean())
            all_roc_curves.append(rocs)
            chance_rates.append(errd[chancek])
        errd_opt = errdls[-1]
        rocs_opt = errd_opt[metric]
        print (lamS_opt, errd_opt[summary_measure].mean())
        all_roc_curves.append(rocs_opt)
        chance_rates.append(errd_opt[chancek])
        toplots = []
        toplots.extend(map(str, lamSs))
        toplots.append(str(lamS_opt))
        linedesc = pd.Series(toplots, name='method')

    pss = []
    for roc_curve_group in all_roc_curves:
        (rs, ps, ts) = fg.pool_roc(roc_curve_group, all_roc_curves, max_x = 1000)
        pss.append(ps)
    
    if normed:
        to_plot = np.dstack(pss) / np.repeat(pss[0][:,:,None], len(pss), axis=2)
    else:
       to_plot = np.dstack(pss)
    xs = pd.Series(rs, name=xl)
    
    sns.tsplot(to_plot, time=xs, condition=linedesc, value=yl)
    plt.hold(True)

    chance_x = [min(xs), max(xs)]
    chance_y = [np.mean(chance_rates), np.mean(chance_rates)]
    #plt.plot(chance_x, chance_y,'--k')
    plt.show()

#plots error dictionaries
#this one includes lots of incorrect orthology
def plot_synthetic_performance2(lamP=1.0, lamR=5, lamSs=[0,1], k=20):
    N = 5
    N_TF = 20
    N_G = 30
    out = os.path.join('data','fake_data','plot_synthetic_performance2')
    ds.write_fake_data1(N1 = k*N, N2 = 5*k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.8, sparse=0.5, fuse_std = 0.0, orth_falsepos=0.5,orth_falseneg=0.5)
    
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamSs))) for m in metrics} #anthracis

    
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False)
        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]

    to_plot = np.dstack((err_dict1['aupr'], err_dict1['aupr_con']))
    linedesc = pd.Series(['full','constrained'],name='error type')
    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='aupr')
    
    plt.show()

#plots error dictionaries
#this one includes lots of incorrect orthology, and solves using em
def plot_synthetic_performance_em(lamP=1.0, lamR=5, lamSs=[0,1], k=20):
    N = 10
    N_TF = 5
    N_G = 30
    out = os.path.join('data','fake_data','plot_synthetic_performance_em')
    ds.write_fake_data1(N1 = k*N, N2 = 5*k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.8, sparse=0.5, fuse_std = 0.0, orth_falsepos=0.75,orth_falseneg=0.0)
    (constraints, marks, orths) = ds.load_constraints(out)

    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamSs))) for m in metrics} #anthracis

    
    for i, lamS in enumerate(lamSs):
        sargs=fr.get_settings({'em_it': 5,'f':1,'uf':1,'marks':marks})
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct_em', reverse = True, cv_both = (True, False), exclude_tfs=False, settings = sargs)
        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]

    to_plot = np.dstack((err_dict1['aupr'], err_dict1['aupr_con']))
    linedesc = pd.Series(['full','constrained'],name='error type')
    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value='aupr')
    
    plt.show()


#plots error dictionaries
#this is really for debugging plot_bacteria_performance 
#uses some priors
def plot_synthetic_performance3(lamP=3.0, lamR=5, lamSs=[0,1], k=20):
    N = 5
    N_TF = 20
    N_G = 30
    out = os.path.join('data','fake_data','plot_synthatic_performance3')
    ds.write_fake_data1(N1 = k*N, N2 = k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.0)
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamSs))) for m in metrics} #anthracis

    
    for i, lamS in enumerate(lamSs):
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False, pct_priors=0.5)
        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]
    print err_dict1
    print err_dict2
    sns.tsplot(err_dict1['aupr'], time=lamSs, legend='full')
    plt.figure()
    sns.tsplot(err_dict1['aupr_con'], time=lamSs, legend='constrained')
    plt.xlabel('lamS')
    plt.ylabel('aupr')
    
    plt.show()


#plots distributions of real and fake beta differences, either for some sample data, or for the data contained in the folder out  
def plot_beta_diffs(out=None, B0=None, B1=None, add_fake=False,scale=1.0):
    N = 10
    N_TF = 1
    N_G = 500
    if out==None:
        out = os.path.join('data','fake_data','beta_diffs_dist')
        ds.write_fake_data1(N1 = N, N2 = N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.5, sparse=0.0, fuse_std = 0.1, orth_falsepos=0.5,orth_falseneg=0.5)

    if B0==None:
        (B0, _, _) = ds.load_network(os.path.join(out, 'beta1'))
    if B1==None:
        (B1, _, _) = ds.load_network(os.path.join(out, 'beta2'))
    (constraints, marks, orths) = ds.load_constraints(out)
    marks = np.array(marks)
    diffs = fr.beta_diff([B0, B1], constraints)
    
    real_diffs =diffs[marks == True] 
    fake_diffs = diffs[marks == False]
    print len(real_diffs)

    if add_fake:
        fakes = np.zeros(real_diffs.shape)
        for i in range(len(real_diffs)/2): #constraints are doubled
            r1 = random.randint(0,B0.shape[0]-1)
            r2 = random.randint(0,B1.shape[0]-1)
            c1 = random.randint(0,B0.shape[1]-1)
            c2 = random.randint(0,B1.shape[1]-1)
            
            diff = B0[r1, c1] - B1[r2, c2]
            fakes[i] = diff
            fakes[i + len(real_diffs)/2] = -diff
        fake_diffs = np.hstack((fake_diffs, fakes))
    
    real_diffs *= scale
    fake_diffs *= scale
    print real_diffs
    print fake_diffs
    #bins = np.linspace(-0.1, 0.1, 100)
    bins = np.linspace(min(fake_diffs), max(fake_diffs), 100)
    if len(real_diffs):
        plt.hist(real_diffs, bins, histtype="stepfilled",alpha=0.5,label='real')
        #sns.kdeplot(real_diffs, shade=True, label='real')
    if len(fake_diffs):
        plt.hist(fake_diffs, bins, histtype="stepfilled",alpha=0.5,label='fake')
        #sns.kdeplot(fake_diffs, shade=True, label='fake')
    plt.xlabel('Delta Beta')
    
        
    plt.legend()
    #plt.figure()
    #plt.hist(diffs[marks==True])
    #print np.std(diffs)
    plt.show()
    

    return (real_diffs, fake_diffs)


#plots distributions of real lambdas of fusion constraints
def plot_fuse_lams(out, cons):

    (constraints, marks, orths) = ds.load_constraints(out)
    marks = np.array(marks)
    lams = np.array(map(lambda x: x.lam, cons))
    print lams
    bins = np.linspace(min(lams), max(lams), 50)
    
    if sum(marks):
        plt.hist(lams[marks==True], bins, histtype="stepfilled",alpha=0.5,label='real')
    
    if sum(marks) < len(marks):
        plt.hist(lams[marks==False], bins, histtype="stepfilled",alpha=0.5,label='fake')
    plt.legend()
    
    plt.show()    




#plots performance as a function of the number of CV folds being used
def synthetic_performance_by_k(lamP=1, lamR=2, lamS=0, ks=[20,15,10,5,2]):

    N = 2
    N_TF = 20
    N_G = 200
    out = os.path.join('data','fake_data','synthetic_performance_by_k')
    k = max(ks)
    ds.write_fake_data1(N1 = k*N, N2 = k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.7, fuse_std = 0.0)
    auprs = np.zeros(len(ks))
    mses = np.zeros(len(ks))
    stes = np.zeros(len(ks))

        
    for i, k in enumerate(ks):
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=0.0, verbose=True)
        aupr = errd1['aupr']
        auprs[i] = aupr.mean()
        stes[i] = np.std(aupr) / k**0.5
        #mse = errd1['mse']
        #mses[i] = mse.mean()
        #stes[i] = np.std(mse) / k**0.5

    #for i in range(len(auprs)):
    #    auprs[i] = auprs[i]/auprs[0]

    ks = 1.0 / np.array(ks)
    pretty_plot_err(ks, auprs, stes, (1, 0.5, 0, 0.25))
    plt.xlabel('fraction of data used')
    plt.ylabel('aupr')
    plt.show()

#plots performance as a function of the number of CV folds being used
def bacteria_performance_by_k(lamP=1, lamR=2, lamS=0, ks=[20,15,10,5,2]):
    out = 'data/bacteria_standard'
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    auprs = np.zeros(len(ks))
    stes = np.zeros(len(ks))
    
    for i, k in enumerate(ks):
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=0.0, verbose=True)
        aupr = errd1['aupr']
        auprs[i] = aupr.mean()
        stes[i] = np.std(aupr) / k**0.5

    ks = 1.0 / np.array(ks)
    pretty_plot_err(ks, auprs, stes, (1, 0.5, 0, 0.25))
    plt.xlabel('fraction of data used')
    plt.ylabel('aupr')
    plt.show()

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


#determines natural variation in solvability of networks 
def net_var():
    out1 = os.path.join('data','fake_data','net_var')
    if not os.path.exists(out1):
        os.mkdir(os.path.join('data','fake_data','net_var'))

    pct_fused = np.arange(0,1,0.1)
    seed = random.random()
    N_TF = 20
    N_G = 200
    k = 2
    lamP = 1
    lamR = 2
    lamS = 0
    reps = 50
    errdict1 = {m: np.zeros((k*reps, 1)) for m in pct_fused}
    errdict2 = {m: np.zeros((k*reps, 1)) for m in pct_fused}


    for fused in pct_fused:
        for i in range(reps):
            out2 = os.path.join(out1,'dat_'+str(fused)+str(i))
            ds.write_fake_data1(out_dir = out2, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1, pct_fused=fused)        
            (errd1, errd2) = fg.cv_model_m(out2, lamP, lamR, lamS, k, solver='solve_ortho_direct', reverse=False, cv_both=(True,False), exclude_tfs=True, seed=seed, verbose=True)
            errdict1[fused][k*i:k*(i+1),:] = errd1['mse']
            errdict2[fused][k*i:k*(i+1),:] = errd2['mse']

    #linedesc = pd.Series(['1','2'], name='network')
    #to_plot = np.dstack((errdict1.values(), errdict2.values()))
    xs = pd.Series(pct_fused, name = 'percent fused')
    sorted_keys = sorted(errdict1.keys())
    to_plot = np.hstack((map(lambda k: errdict1[k], sorted_keys)))
    sns.tsplot(to_plot, time=xs, value='mse')
    plt.show()

def pretty_plot_err(x, y, errbar, color=(1,0,0,1)):
    plt.plot(x, y, color=color)
    plt.fill_between(x, y-errbar, y+errbar, color=color)



# shows penalty weight, fusion penalty, and ddx fusion penalty for an arbitrary pair of variances under EM penalty
def em_sillytest():
    var1 = 1.0
    var2 = 20.0

    delta_bs = np.linspace(0,5,100)
    
    p1 = scipy.stats.norm.pdf(delta_bs, loc=0, scale=var1**0.5)
    p2 = scipy.stats.norm.pdf(delta_bs, loc=0, scale=var2**0.5)
    

    lam1 = p1 / (p1 + p2) * (1/var1)
    lam2 =  p2 / (p1 + p2) * (1/var2)
    pen1 = lam1 * delta_bs**2
    pen2 = lam2 * delta_bs**2

    pen = pen1 + pen2
    lam = lam1 + lam2
    plt.subplot(311)
    plt.plot(delta_bs, lam)
    plt.xlabel('delta beta')
    plt.ylabel('fusion penalty weight')
    #plt.show()

    plt.subplot(312)
    plt.plot(delta_bs, pen)
    plt.xlabel('delta beta')
    plt.ylabel('fusion penalty')
    #plt.show()

    plt.subplot(313)
    plt.plot(delta_bs[0:-1], np.diff(pen))
    plt.xlabel('delta beta')
    plt.ylabel('derivative of penalty')
    plt.show()

#uses data from test_mcp3; plots lamS for scad and mcp  for true and false orths
#running off of true networks, not estimated networks--need to fix
def plot_lamS():
    a = 0.5
    lamS = 3
    out = os.path.join('data','fake_data','test_mcp3','dat_1')
    (constraints, marks, orth) = ds.load_constraints(out)
    (b1, genes1, tfs1) = ds.load_network(os.path.join(out, 'beta1'))
    (b2, genes2, tfs2) = ds.load_network(os.path.join(out, 'beta2'))
#    con_val = np.array(map(lambda x: x.lam, constraints))
    true_cons = []
    false_cons = []

    for i in range(len(constraints)):
        if marks[i] == 1:
            true_cons.append(constraints[i])
        else:
            false_cons.append(constraints[i])

    scad_cons_t = fr.scad([b1, b2], true_cons, lamS, a=a)
    scad_t = np.array(map(lambda x: x.lam, scad_cons_t))
    scad_cons_f = fr.scad([b1, b2], false_cons, lamS, a=a)
    scad_f = np.array(map(lambda x: x.lam, scad_cons_f))

    mcp_cons_t = fr.mcp([b1, b2], true_cons, lamS, a=0.2)
    mcp_t = np.array(map(lambda x: x.lam, mcp_cons_t))
    mcp_cons_f = fr.mcp([b1, b2], false_cons, lamS, a=0.2)
    mcp_f = np.array(map(lambda x: x.lam, mcp_cons_f))

    return (scad_t, scad_f, mcp_t, mcp_f)
    plt.hist(scad_con_val)
    plt.show()
    plt.hist(mcp_con_val)
    plt.show()

#uses data from test_mcp3; plots lamS for scad and mcp  for true and false orths
def plot_lamS_real():
    a = 0.5
    lamS = 3
    orth_falsepos = 0.3
    orth_falseneg = 0
    lamP = 1
    lamR = 10

    out = 'data/bacteria_standard'
    subt = ds.standard_source('data/bacteria_standard',0)
    anthr = ds.standard_source('data/bacteria_standard',1)
    (bs_e, bs_t, bs_genes, bs_tfs) = subt.load_data()
    (ba_e, ba_t, ba_genes, ba_tfs) = anthr.load_data()
    organisms = [subt.name, anthr.name]
    gene_ls = [bs_genes, ba_genes]
    tf_ls = [bs_tfs, ba_tfs]
    Xs = [bs_t, ba_t]
    Ys = [bs_e, ba_e]
    (bs_priors, bs_sign) = subt.get_priors()
    (ba_priors, ba_sign) = anthr.get_priors()
    priors = bs_priors + ba_priors
    
    orths = ds.load_orth('data/bacteria_standard/orth',[subt.name, anthr.name])
    orth = ds.generate_faulty_orth(orths, bs_genes, bs_tfs, ba_genes, ba_tfs, organisms, orth_falsepos, orth_falseneg)
    (constraints, marks) = fr.orth_to_constraints_marked(organisms, gene_ls, tf_ls, orth, lamS)
    
    Bs = fr.solve_ortho_direct(organisms, gene_ls, tf_ls, Xs, Ys, orth, priors, lamP, lamR, lamS)
    b1 = Bs[0]
    b2 = Bs[1]

    true_cons = []
    false_cons = []

    for i in range(len(constraints)):
        if marks[i] == 1:
            true_cons.append(constraints[i])
        else:
            false_cons.append(constraints[i])

    scad_cons_t = fr.scad([b1, b2], true_cons, lamS, a=a)
    scad_t = np.array(map(lambda x: x.lam, scad_cons_t))
    scad_cons_f = fr.scad([b1, b2], false_cons, lamS, a=a)
    scad_f = np.array(map(lambda x: x.lam, scad_cons_f))

    mcp_cons_t = fr.mcp([b1, b2], true_cons, lamS, a=0.2)
    mcp_t = np.array(map(lambda x: x.lam, mcp_cons_t))
    mcp_cons_f = fr.mcp([b1, b2], false_cons, lamS, a=0.2)
    mcp_f = np.array(map(lambda x: x.lam, mcp_cons_f))

    return (scad_t, scad_f, mcp_t, mcp_f)
    plt.hist(scad_con_val)
    plt.show()
    plt.hist(mcp_con_val)
    plt.show()

def show_penalty():
    lamS = 1
    a=1.0
    def miniscad_dx(theta):
        theta = np.abs(theta)
        if theta < a/2:
            return theta * lamS
        if theta >= a/2:
            return lamS * max(0, (a - theta))
    def miniL2dx(theta):
        return theta*lamS

    xs = np.linspace(0,1.5,100)
    dms = map(miniscad_dx, xs)
    dl2 = map(miniL2dx, xs)
    plt.plot(xs, dms)
    plt.plot(xs, dl2,'--')
    plt.legend(('SCAD-like penalty','L2'))
    plt.xlabel('|B0 - B1|')
    plt.ylabel('derivative of penalty')
    plt.plot([a/2,a/2],[0,0.3],color=(0.5,0.5,0.5))
    plt.show()
    plt.plot(xs, (xs[1]-xs[0])*np.cumsum(dms))
    #plt.show()
    plt.plot(xs, (xs[1]-xs[0])*np.cumsum(dl2),'--')
    
    plt.legend(('SCAD-like penalty', 'L2'))
    plt.xlabel('|B0 - B1|')
    plt.ylabel('penalty')
    plt.show()

    plt.plot(xs, (xs[1]-xs[0])*np.cumsum(dms))
    #plt.show()
    plt.plot(xs, (xs[1]-xs[0])*np.cumsum(dl2),'--')
    plt.plot([a/2,a/2],[0,0.3],color=(0.5,0.5,0.5))
    plt.legend(('SCAD-like penalty', 'L2'))
    plt.xlabel('|B0 - B1|')
    plt.ylabel('penalty')
    plt.show()



#makes sure that XB = Y for generated data

def verify_data_integrity(out=None,N=100, N_TF=10, N_G=10):
    if out==None:
        out = os.path.join('data','fake_data','verify_data_integrity')
        ds.write_fake_data1(N1=N, N2=N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.0, measure_noise2 = 0.0, sparse=0.0, fuse_std = 0.0, pct_fused=1.0)
    
    (n1, g1, t1) = ds.load_network(os.path.join(out, 'beta1'))
    (n2, g2, t2) = ds.load_network(os.path.join(out, 'beta2'))

    d1 = ds.standard_source(out, 0)
    d2 = ds.standard_source(out, 1)

    err1 = ((np.dot(d1.tf_mat, n1) - d1.exp_mat)**2).mean()
    err2 = ((np.dot(d2.tf_mat, n2) - d2.exp_mat)**2).mean()
    print err1
    print err2


#plots error dictionaries
#this is really for debugging plot_bacteria_performance 
def plot_synthetic_performance_adj(lamP=1.0, lamR=5, lamSs=[0,1], k=20):
    N = 5
    N_TF = 20
    N_G = 30

    out = os.path.join('data','fake_data','plot_synthetic_performance')
    ds.write_fake_data1(N1 = k*N, N2 = k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.5, sparse=0.5, fuse_std = 0.0, orth_falsepos=10.0,orth_falseneg=0.0)
    
    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con','B_mse']
    err_dict_a = {m : np.zeros((k, len(lamSs))) for m in metrics} #subtilis
    err_dict_u = {m : np.zeros((k, len(lamSs))) for m in metrics} #anthracis
    settings_adj = fr.get_settings({'adjust':True,'return_cons':True})
    for i, lamS in enumerate(lamSs):
        (errd1a, errd2a) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False, settings = settings_adj)
        (errd1u, errd2u) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse = True, cv_both = (True, False), exclude_tfs=False, settings = None)
        
        print 'corr is %f' % errd1u['corr'].mean()
        for metric in metrics:
            err_dict_a[metric][:, [i]] = errd1a[metric]
            err_dict_u[metric][:, [i]] = errd1u[metric]

    measures = ('auc','auc')
    to_plot = np.dstack((err_dict_a[measures[0]], err_dict_u[measures[1]]))
    linedesc = pd.Series(('adjusted','not'),name='error type')
    xs = pd.Series(lamSs, name='lamS')
    sns.tsplot(to_plot, time=xs, condition=linedesc, value=measures[0])
    
    plt.show()


#plots performance as a function of R
def plot_synthetic_performanceR(lamP=1.0, lamRs=[1,4,7,10,13], lamS=0, k=20):
    N = 3
    N_TF = 20
    N_G = 30
    out = 'data/fake_data/syntheticR'
    ds.write_fake_data1(N1 = k*N, N2 = k*N, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1,pct_fused=0.5, sparse=0.5, fuse_std = 0.0, orth_falsepos=10.0,orth_falseneg=0.0)

    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con']
    err_dict1 = {m : np.zeros((k, len(lamRs))) for m in metrics} #subtilis
    err_dict2 = {m : np.zeros((k, len(lamRs))) for m in metrics} #anthracis

    rseed = random.random()
    for i, lamR in enumerate(lamRs):
        (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=k, solver='solve_ortho_direct', reverse=True, cv_both=(True,False), exclude_tfs=False, pct_priors=0.0, verbose=False, seed=rseed)

        for metric in metrics:
            err_dict1[metric][:, [i]] = errd1[metric]
            err_dict2[metric][:, [i]] = errd2[metric]
    metric = 'mse'
    errs = err_dict1[metric]

    stes = errs.std(axis=0) / errs.shape[0]**0.5
    pretty_plot_err(lamRs, errs.mean(axis=0), stes, (1, 0.5, 0, 0.25))
    plt.show()


def scad_priors():
    N_TF = 20
    N_G = 200
    amt_fused = 0.5
    lamPs = [1.0,0.0001,0.000001]
    lamS = 2
    lamR = 4
    seed = 10
    reps = 1
    k = 5

    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con','B_mse']
    err_dict = {m : np.zeros((k, len(lamPs))) for m in metrics}

    if not os.path.exists(os.path.join('data','fake_data','scad_priors')):
        os.mkdir(os.path.join('data','fake_data','scad_priors'))
    out = os.path.join('data','fake_data','scad_priors')

    for p in range(reps):
        for i, P in enumerate(lamPs):
            ds.write_fake_data1(N1 = 5*3, N2 = 5*20, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1, orth_falsepos = 0.5)

            (errd1l, errd2l) = fg.cv_model_m(out, lamP=P, lamR=lamR, lamS=lamS, k=5, solver='solve_ortho_direct', reverse = False, cv_both = (True, True), pct_priors=0.7)
            for metric in metrics:
                err_dict[metric][:, [i]] = errd1l[metric]

    return (err_dict)


#returns a list of the top k (pre-specified when running model) interactions that are in the prior, appear in network 1, but not in network 2. and vice versa
def top_k_diff(top_k_1, labels1, top_k_2, labels2, k):
    top_k_1 = top_k_1[0:k]
    top_k_2 = top_k_2[0:k]
    top_k_true_1 = map(lambda i: top_k_1[i], filter(lambda j: labels1[j] == 1, range(len(top_k_1))))
    top_k_true_2 = map(lambda i: top_k_2[i], filter(lambda j: labels2[j] == 1, range(len(top_k_2))))

    #top_k_true_1 = top_k_true_1[0:k]
    #top_k_true_2 = top_k_true_2[0:k]

    top_k_true_s_2 = set(top_k_true_2)
    top_k_true_s_1 = set(top_k_true_1)

    top_k_true_excl_1 = filter(lambda tki: not tki in top_k_true_s_2, top_k_true_1)
    top_k_true_excl_2 = filter(lambda tki: not tki in top_k_true_s_1, top_k_true_2)

    return (top_k_true_excl_1, top_k_true_excl_2)



#compares performance on constrained and non-constrained portions of the network
def con_noncon():
    import copy
    N_TF = 20
    N_G = 200
    amt_fuseds = [0,0.25,0.5,0.75]
    lamSs = [0,2,4]
    seed = 10
    k = 5

    metrics = ['mse','R2','aupr','auc','corr', 'auc_con','aupr_con','auc_noncon', 'aupr_noncon', 'B_mse']
    err_dict_l = {m : np.zeros((k, len(lamSs))) for m in metrics}
    err_l = {f : copy.deepcopy(err_dict_l) for f in amt_fuseds}

    if not os.path.exists(os.path.join('data','fake_data','con_noncon')):
        os.mkdir(os.path.join('data','fake_data','non_con'))
    #iterate over how much fusion

    for f, amt_fused in enumerate(amt_fuseds):
        out1 = os.path.join('data','fake_data','con_noncon','dat_'+str(f))
        if not os.path.exists(out1):
            os.mkdir(os.path.join('data','fake_data','con_noncon','dat_'+str(f)))
        
        ds.write_fake_data1(N1 = 3*5, N2 = 3*50, out_dir = out1, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1)
        lamR = 2
        lamP = 1.0 #priors don't matter

        for j, lamS in enumerate(lamSs):
            (errd1l, errd2l) = fg.cv_model_m(out1, lamP=lamP, lamR=lamR, lamS=lamS, k=5, solver='solve_ortho_direct', reverse = False, cv_both = (True, True))
            for metric in metrics:
                err_l[amt_fused][metric][:, [j]] = errd1l[metric]

    o0 = np.dstack((err_l[0]['aupr_con'], err_l[0]['aupr_noncon'], err_l[0]['aupr']))
    o025 = np.dstack((err_l[0.25]['aupr_con'], err_l[0.25]['aupr_noncon'], err_l[0.25]['aupr']))
    o05 = np.dstack((err_l[0.5]['aupr_con'], err_l[0.5]['aupr_noncon'], err_l[0.5]['aupr']))
    o075 = np.dstack((err_l[0.75]['aupr_con'], err_l[0.75]['aupr_noncon'], err_l[0.75]['aupr']))

    xax = pd.Series(lamSs, name="lamS")
    conds = pd.Series(["Constrained", "Non constrained", "Whole network"], name="method")

    plt.subplot(141)
    sns.tsplot(o0, time=xax, condition=conds, value="AUPR")
    plt.axis([0,4,0.65,1])
    plt.title("Amount fused = 0")

    plt.subplot(142)
    sns.tsplot(o025, time=xax, condition=conds, value="AUPR")
    plt.axis([0,4,0.65,1])
    plt.title("Amount fused = 0.25")

    plt.subplot(143)
    sns.tsplot(o05, time=xax, condition=conds, value="AUPR")
    plt.axis([0,4,0.65,1])
    plt.title("Amount fused = 0.5")

    plt.subplot(144)
    sns.tsplot(o075, time=xax, condition=conds, value="AUPR")
    plt.axis([0,4,0.65,1])
    plt.title("Amount fused = 0.75")
    plt.show()

    return (err_dict_l)


def lamS_dist_fig():

    N_TF = 20
    N_G = 200
    amt_fused = 0.5
    orth_err = [0.75]
    lamSs = [4]#[0, 2, 4, 6]
    seed = 10
    k = 5


    if not os.path.exists(os.path.join('data','fake_data','lamS_dist_fig')):
        os.mkdir(os.path.join('data','fake_data','lamS_dist_fig'))
    #iterate over how much fusion

    for i, N in enumerate(orth_err):
        out = os.path.join('data','fake_data','lamS_dist_fig','dat_'+str(N))
        ds.write_fake_data1(N1 = 5*70, N2 = 570, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = amt_fused, orth_falsepos = N, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.5, fuse_std = 0.1)
        lamR = 2
        lamP = 1.0 #priors don't matter

        for j, lamS in enumerate(lamSs):

            scad_settings = fr.get_settings({'s_it':20, 'per':((1/(1+float(N)))*100), 'return_cons' : True})
            (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS=lamS, k=5, solver='solve_ortho_direct_scad', reverse = False, settings = scad_settings, cv_both = (True, True))
            print scad_settings['cons']
            plot_fuse_lams(out, scad_settings['cons'])

#looks at how many operon constraints fall both in or out of priors
def check_operons_priors():
    d1 = ds.standard_source('data/bacteria_standard', 0)
    (con, marks, orth) = ds.load_constraints('data/bacteria_standard','operon')
    (priors, signs) = d1.get_priors()
    priors_s = set(priors)
    def coeff_to_g(coeff):
        tf = fr.one_gene(name = d1.tfs[coeff.r], organism = d1.name)
        gene = fr.one_gene(name = d1.genes[coeff.c], organism = d1.name)
        return (tf, gene)
    con_pairs = map(lambda c: (coeff_to_g(c.c1), coeff_to_g(c.c2)), con)

    both = filter(lambda cp: cp[0] in priors_s and cp[1] in priors_s, con_pairs)
    neither = filter(lambda cp: not (cp[0] in priors_s or cp[1] in priors_s), con_pairs)
    chance_in_priors = float(len(priors)) / (len(d1.tfs) * len(d1.genes))
    chance_both = chance_in_priors ** 2

    chance_neither = (1-chance_in_priors)**2

    print (float(len(both))/len(con_pairs), float(len(neither))/len(con_pairs))

    print (chance_both, chance_neither)

    print (float(len(both))/len(con_pairs) / chance_both)

def try_solve_operon(lamP = 1.0, lamR = 1.0, lamS = 2):
    settings = fr.get_settings()
    solver = 'solve_ortho_direct'    
    out = 'data/bacteria_standard'
    seed = 5
    (errd1, errd2) = fg.cv_model_m(out, lamP=lamP, lamR=lamR, lamS =lamS, k=1, settings = settings, reverse = True, cv_both = (True, False), exclude_tfs=False, seed = seed, orth_file='operon')
    return errd1


#as plot synthetic performance, except it plots overlaid ROC curves for different values of lamS
def plot_operon_roc(lamP=1.0, lamR=5, lamSs=[0,1], k=20, metric='roc',savef=None,normed=False,scad=False):
    out = os.path.join('data','bacteria_standard')
    plot_roc(out, lamP, lamR, lamSs, k, metric, savef=savef,normed=normed,scad=scad, orth_file='operon')


#tests to see if the iterative solver is getting the same solution as the non iterative solver on a small network
def test_iter1():
    N_TF = 5
    N_G = 10
    N1 = 4
    N2 = 4
    out = os.path.join('data','fake_data','iter_test1')
    ds.write_fake_data1(N1 = N1, N2 = N2, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.0)
    
    (b1n, b2n) = fg.fit_model(out, 1.0, 0.1, 0.0, solver='solve_ortho_direct')
    (b1d, b2d) = fg.fit_model(out, 1.0, 0.1, 0.5, solver='solve_ortho_direct')
    si = fr.get_settings({'it':1000})
    (b1i, b2i) = fg.fit_model(out, 1.0, 0.1, 0.5, solver='iter_solve', settings=si)
    return (b1n, b1d, b1i)

def test_iter2(start,stop,num,reps):
    N_TF = 5
    N_G = 10
    N1 = 7
    N2 = 7
    out = os.path.join('data','fake_data','iter_test2')
    ds.write_fake_data1(N1 = N1, N2 = N2, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.0)

    its = np.linspace(start,stop,num)
    iterdiff = []
    noniterdiff = []
    for i in range(len(its)):
        iterdiff.append(np.zeros((reps)))
        noniterdiff.append(np.zeros((reps)))
    
    for j in range(reps):
        for i in range(len(its)):
            (b1n, b2n) = fg.fit_model(out, 1.0, 0.0, 0.0, solver='solve_ortho_direct')
            (b1d, b2d) = fg.fit_model(out, 1.0, 0.0, 0.5, solver='solve_ortho_direct')
            si = fr.get_settings({'it':int(its[i])})
            (b1i, b2i) = fg.fit_model(out, 1.0, 0.0, 0.5, solver='iter_solve', settings=si)
            iterdiff[i][j] = abs(b1d-b1i).sum()
            #print iterdiff[i][j]
            noniterdiff[i][j] = abs(b1n-b1i).sum()
            #print noniterdiff[i][j]
    toplot = np.dstack((iterdiff, noniterdiff))
    print b1n
    print '-------------'
    print b1d
    print '-------------'
    print b1i
    plt.matshow(b1d)
    plt.show(block=False)
    plt.matshow(b1i)
    plt.show(block=False)
    return b1i
    return toplot	
    xax = pd.Series(its, name="iterations")
    conds = pd.Series(["iter-direct fused", "iter-direct unfused"], name="type")
    sns.tsplot(toplot, time=xax, condition=conds)
    plt.axis(its)

    

