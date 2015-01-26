import fused_reg as fr
import data_sources as ds
import fit_grn as fg
import numpy as np
import os
from matplotlib import pyplot as plt
#This file is just a list of experiments. Only code that is used nowhere else is appropriate (ie general plotting code should go somewhere else


#In this experiment, we generate several data sets with a moderate number of TFs and genes. Fusion is total and noiseless. Measure performance as a function of lamS, and performance as a function of the amount of data. Plot both. This is a basic sanity check for fusion helping
def sanity1():
    repeats = 10
    N_TF = 25
    N_G = 1000
    data_amnts = [10, 20,30,40]
    lamSs = [0, 0.25,0.5,0.75,1]
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
                errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True)
                errors[i, j] += errd['mse'][0]
    for i in range(len(len(data_amnts))):
        for j in range(len(lamSs)):
            errors[i,j] == errors[i,j]/k
    plt.plot(errors.T)
    plt.legend(data_amnts)
    plt.savefig(os.path.join(os.path.join('data','fake_data','sanity1','fig1')))
    plt.figure()
    plt.plot(errors)
    plt.legend(lamSs)
    plt.savefig(os.path.join(os.path.join('data','fake_data','sanity1','fig2')))

def increase_data():
    repeats = 10
    N_TF = 25
    N_G = 1000
    fixed_data_amt = 20
    data_amnts = [10, 20,30,40]
    lamSs = [0, 0.25,0.5,0.75,1]
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
                errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct', reverse = True)
                errors[i, j] += errd['mse'][0]
    for i in range(len(len(data_amnts))):
        for j in range(len(lamSs)):
            errors[i,j] == errors[i,j]/k
    plt.plot(errors.T)
    plt.legend(data_amnts)
    plt.savefig(os.path.join(os.path.join('data','fake_data','increasedata','fig1')))
    plt.figure()
    plt.plot(errors)
    plt.legend(lamSs)
    plt.savefig(os.path.join(os.path.join('data','fake_data','increasedata','fig2')))


def test_scad():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    N_TF = 25
    N_G = 1000
    amt_fused = [0, 0.25,0.5,0.75,1]
    lamSs = [0, 0.25,0.5,0.75,1]
    if not os.path.exists(os.path.join('data','fake_data','test_scad')):
        os.mkdir(os.path.join('data','fake_data','test_scad'))
    #iterate over how much fusion
    errors = np.zeros((len(amt_fused), len(lamSs)))
    for i, N in enumerate(amt_fused):
        out = os.path.join('data','fake_data','test_scad','dat_'+str(N))
        ds.write_fake_data1(N1 = 30, N2 = 30, out_dir = out, tfg_count1=(N_TF, N_G), tfg_count2 = (N_TF, N_G), pct_fused = N, orth_falsepos = 0.3, orth_falseneg = 0.3, measure_noise1 = 0.1, measure_noise2 = 0.1, sparse=0.0, fuse_std = 0.1)
        lamR = 0.1
        lamP = 1.0 #priors don't matter
        for j, lamS in enumerate(lamSs):
            errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=lamS, k=10, solver='solve_ortho_direct_scad', reverse = True)
            errors[i,j] = errd['mse'][0]

    plt.matshow(errors.T)
    plt.savefig(os.path.join(os.path.join('data','fake_data','test_scad','fig1')))
    plt.figure()



def studentseminar():
#create simulated data set with false orthology and run fused L2 and fused scad 
     
    lamSs = [0, 0.25,0.5,0.75,1]
    errors = np.zeros(len(lamSs))
    for i, N in enumerate(lamSs):
        out = os.path.join('data','bacteria_standard')
        lamR = 0.1
        lamP = 1.0 #priors don't matter
        errd = fg.cv_model1(out, lamP=lamP, lamR=lamR, lamS=N, k=10, solver='solve_ortho_direct', reverse = True)
        errors[i] = errd['aupr'][0]
    
    plt.plot(lamSs, errors, 'ro')
    plt.savefig(os.path.join(os.path.join('data','bacteria_standard','studentseminar','fig1')))
    plt.figure()
