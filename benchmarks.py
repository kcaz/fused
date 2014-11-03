import bacteria as ba
import numpy as np
import random
import fused_L2 as fl
import scipy.optimize

def single_species_benchmarks():
    repeats = 2
    lamPs = np.array([0])#np.logspace(0, 3, 10)
    lamSs = np.array([0])
    lamRs = np.logspace(0, 3, 5)
    iron_conds = range(ba.iron_conds)
    timeseries_conds = range(ba.timeseries_conds)
    subtilis_conds = range(ba.subtilis_conds)
    outf = file('benchmark_results/single_species_bench','w')
    for lamP in lamPs:
        for lamS in lamSs:
            for lamR in lamRs:
                
                acc_s = 0
                acc_a = 0
                for rep in range(repeats):
                    random.shuffle(iron_conds)
                    random.shuffle(timeseries_conds)
                    random.shuffle(subtilis_conds)

                    sub_s = subtilis_conds[0:int(len(subtilis_conds)/2)]
                    sub_i = iron_conds[0:int(len(iron_conds)/2)]
                    sub_t = timeseries_conds[0:int(len(timeseries_conds)/2)]
                    
                    ba.run_both(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp', sub_s = sub_s, sub_i = sub_i, sub_t = sub_t)
                    
                    sub_st = subtilis_conds[int(len(subtilis_conds)/2):len(subtilis_conds)]
                    sub_it = iron_conds[int(len(iron_conds)/2):len(iron_conds)]
                    sub_tt = timeseries_conds[int(len(timeseries_conds)/2):len(timeseries_conds)]                    

                    (bs_e, bs_t, bs_genes, bs_tfs) = ba.load_B_subtilis(sub_st)
                    (ba_e, ba_t, ba_genes, ba_tfs) = ba.load_B_anthracis(sub_it, sub_tt)

                    eval_s = ba.eval_prediction('tmp_subtilis', bs_e, bs_t, bs_genes, bs_tfs,'R2')
                    eval_a = ba.eval_prediction('tmp_anthracis', ba_e, ba_t, ba_genes, ba_tfs,'R2')
                    acc_s += eval_s
                    acc_a += eval_a
                writestr = 'lamP=%f\tlamS=%f\tlamR=%f\tsubtilis=%f\tanthracis=%f\n' % (lamP, lamS, lamR, acc_s/repeats, acc_a/repeats)
                outf.write(writestr)
                print writestr
                    
    outf.close()


def multi_species_benchmarks():
    repeats = 2
    lamPs = np.array([1])#np.logspace(0, 3, 10)
    lamSs = np.logspace(0, 2, 5)
    lamRs = np.logspace(0, 2, 5)
    iron_conds = range(ba.iron_conds)
    timeseries_conds = range(ba.timeseries_conds)
    subtilis_conds = range(ba.subtilis_conds)
    outf = file('benchmark_results/multi_species_bench','w')
    for lamP in lamPs:
        for lamS in lamSs:
            for lamR in lamRs:
                
                acc_s = 0
                acc_a = 0
                for rep in range(repeats):
                    random.shuffle(iron_conds)
                    random.shuffle(timeseries_conds)
                    random.shuffle(subtilis_conds)

                    sub_s = subtilis_conds[0:int(len(subtilis_conds)/2)]
                    sub_i = iron_conds[0:int(len(iron_conds)/2)]
                    sub_t = timeseries_conds[0:int(len(timeseries_conds)/2)]
                    
                    ba.run_both(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp', sub_s = sub_s, sub_i = sub_i, sub_t = sub_t)
                    
                    sub_st = subtilis_conds[int(len(subtilis_conds)/2):len(subtilis_conds)]
                    sub_it = iron_conds[int(len(iron_conds)/2):len(iron_conds)]
                    sub_tt = timeseries_conds[int(len(timeseries_conds)/2):len(timeseries_conds)]                    

                    (bs_e, bs_t, bs_genes, bs_tfs) = ba.load_B_subtilis(sub_st)
                    (ba_e, ba_t, ba_genes, ba_tfs) = ba.load_B_anthracis(sub_it, sub_tt)

                    eval_s = ba.eval_prediction('tmp_subtilis', bs_e, bs_t, bs_genes, bs_tfs,'R2')
                    eval_a = ba.eval_prediction('tmp_anthracis', ba_e, ba_t, ba_genes, ba_tfs,'R2')
                    acc_s += eval_s
                    acc_a += eval_a
                writestr = 'lamP=%f\tlamS=%f\tlamR=%f\tsubtilis=%f\tanthracis=%f\n' % (lamP, lamS, lamR, acc_s/repeats, acc_a/repeats)
                outf.write(writestr)
                print writestr
                    
    outf.close()

def multi_species_benchmarks2(solver):
    repeats = 50
    lamPs = np.array([1])#np.logspace(0, 3, 10)
    lamSs = [0]#+list(np.logspace(-1.5, 1.5, 3))
    lamRs = [3]#np.logspace(0, 2, 5)
    iron_conds = range(ba.iron_conds)
    timeseries_conds = range(ba.timeseries_conds)
    subtilis_conds = range(ba.subtilis_conds)
    outf = file('benchmark_results/multi_species_bench4','w')
    (BS_priors, sign) = ba.load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    orth = ba.load_orth('bs_ba_ortho_804',['B_anthracis','B_subtilis']) #TODO make the order impossible to get 
    
    for lamP in lamPs:
        for lamS in lamSs:
            for lamR in lamRs:
                
                acc_s = 0
                acc_a = 0
                aupr = 0
                corr = 0
                for rep in range(repeats):
                    random.shuffle(iron_conds)
                    random.shuffle(timeseries_conds)
                    random.shuffle(subtilis_conds)

                    sub_s_i = int(len(subtilis_conds)/5)
                    sub_i_i = int(len(iron_conds)*0.75)
                    sub_t_i = int(len(timeseries_conds)*0.75)

                    sub_s = subtilis_conds[0:sub_s_i]
                    sub_i = iron_conds[0:sub_i_i]
                    sub_t = timeseries_conds[0:sub_t_i]
                    if solver=='run_both':
                        ba.run_both(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp2', sub_s = sub_s, sub_i = sub_i, sub_t = sub_t)
                    if solver=='run_both_adjust':
                        ba.run_both_adjust(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp2', sub_s=sub_s, sub_i=sub_i, sub_t=sub_t)
                    if solver=='run_both_refit':
                        ba.run_both_refit(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp2', sub_s = sub_s, sub_i = sub_i, sub_t = sub_t, it=10,k=20)
                    if solver=='scad':
                        ba.run_both(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp2', sub_s=sub_s, sub_i=sub_i, sub_t=sub_t, it=10, k=20, it_s=10)

                    sub_st = subtilis_conds[sub_s_i:]
                    sub_it = iron_conds[sub_i_i:]
                    sub_tt = timeseries_conds[sub_t_i:]                    

                    (bs_e, bs_t, bs_genes, bs_tfs) = ba.load_B_subtilis(sub_st)
                    (ba_e, ba_t, ba_genes, ba_tfs) = ba.load_B_anthracis(sub_it, sub_tt)

                    corr += ba.betas_fused_corr('tmp2_subtilis', 'tmp2_anthracis', orth)
                    print corr/rep
                    eval_s = ba.eval_prediction('tmp2_subtilis', bs_e, bs_t, bs_genes, bs_tfs,'R2')
                    eval_a = ba.eval_prediction('tmp2_anthracis', ba_e, ba_t, ba_genes, ba_tfs,'R2')

                    (net, bs_genes, bs_tfs) = ba.load_network('tmp2_subtilis')
                    aupr += ba.eval_network_pr(net, bs_genes, bs_tfs, BS_priors)
                    print aupr/rep
                    acc_s += eval_s
                    acc_a += eval_a
                writestr = 'lamP=%f\tlamS=%f\tlamR=%f\tsubtilis=%f\tanthracis=%f\taupr=%f\tcorr_bs=%f\n' % (lamP, lamS, lamR, acc_s/repeats, acc_a/repeats, aupr/repeats, corr/repeats)
                outf.write(writestr)
                print writestr
                    
    outf.close()


def multi_species_benchmarks_opt1(lamSR):
    repeats = 50
    lamP = 1
    #lamSs = [0.3,0.5,0.7,1]#+list(np.logspace(-1.5, 1.5, 3))
    #lamRs = [3,5,7]#np.logspace(0, 2, 5)
    iron_conds = range(ba.iron_conds)
    timeseries_conds = range(ba.timeseries_conds)
    subtilis_conds = range(ba.subtilis_conds)
    (BS_priors, sign) = ba.load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    orth = ba.load_orth('bs_ba_ortho_804',['B_anthracis','B_subtilis']) #TODO make the order impossible to get 

    lamR = lamSR[1]
    lamS = lamSR[0]
  
    acc_s = 0
    acc_a = 0
    aupr = 0
    corr = 0
    for rep in range(repeats):
        random.shuffle(iron_conds)
        random.shuffle(timeseries_conds)
        random.shuffle(subtilis_conds)

        sub_s_i = int(len(subtilis_conds)/8)
        sub_i_i = int(len(iron_conds)*0.75)
        sub_t_i = int(len(timeseries_conds)*0.75)

        sub_s = subtilis_conds[0:sub_s_i]
        sub_i = iron_conds[0:sub_i_i]
        sub_t = timeseries_conds[0:sub_t_i]
        
        ba.run_both_adjust(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp2', sub_s=sub_s, sub_i=sub_i, sub_t=sub_t)

        sub_st = subtilis_conds[sub_s_i:]
        sub_it = iron_conds[sub_i_i:]
        sub_tt = timeseries_conds[sub_t_i:]                    

        (bs_e, bs_t, bs_genes, bs_tfs) = ba.load_B_subtilis(sub_st)
        (ba_e, ba_t, ba_genes, ba_tfs) = ba.load_B_anthracis(sub_it, sub_tt)

        corr += ba.betas_fused_corr('tmp2_subtilis', 'tmp2_anthracis', orth)
        print corr/(rep+1)
        eval_s = ba.eval_prediction('tmp2_subtilis', bs_e, bs_t, bs_genes, bs_tfs,'R2')
        eval_a = ba.eval_prediction('tmp2_anthracis', ba_e, ba_t, ba_genes, ba_tfs,'R2')

        (net, bs_genes, bs_tfs) = ba.load_network('tmp2_subtilis')
        aupr += ba.eval_network_pr(net, bs_genes, bs_tfs, BS_priors)
        print aupr/(rep+1)
        acc_s += eval_s
        acc_a += eval_a
    return aupr/repeats


def multi_species_benchmarks_opt2(lamR):
    repeats = 50
    lamP = 1
    #lamSs = [0.3,0.5,0.7,1]#+list(np.logspace(-1.5, 1.5, 3))
    #lamRs = [3,5,7]#np.logspace(0, 2, 5)
    iron_conds = range(ba.iron_conds)
    timeseries_conds = range(ba.timeseries_conds)
    subtilis_conds = range(ba.subtilis_conds)
    (BS_priors, sign) = ba.load_priors('gsSDnamesWithActivitySign082213','B_subtilis')
    orth = ba.load_orth('bs_ba_ortho_804',['B_anthracis','B_subtilis']) #TODO make the order impossible to get 

    lamR = lamR[0]
    lamS = 0
  
    acc_s = 0
    acc_a = 0
    aupr = 0
    corr = 0
    for rep in range(repeats):
        random.shuffle(iron_conds)
        random.shuffle(timeseries_conds)
        random.shuffle(subtilis_conds)

        sub_s_i = int(len(subtilis_conds)/8)
        sub_i_i = int(len(iron_conds)*0.75)
        sub_t_i = int(len(timeseries_conds)*0.75)

        sub_s = subtilis_conds[0:sub_s_i]
        sub_i = iron_conds[0:sub_i_i]
        sub_t = timeseries_conds[0:sub_t_i]
        
        ba.run_both_adjust(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp2', sub_s=sub_s, sub_i=sub_i, sub_t=sub_t)

        sub_st = subtilis_conds[sub_s_i:]
        sub_it = iron_conds[sub_i_i:]
        sub_tt = timeseries_conds[sub_t_i:]                    

        (bs_e, bs_t, bs_genes, bs_tfs) = ba.load_B_subtilis(sub_st)
        (ba_e, ba_t, ba_genes, ba_tfs) = ba.load_B_anthracis(sub_it, sub_tt)

        corr += ba.betas_fused_corr('tmp2_subtilis', 'tmp2_anthracis', orth)
        print corr/(rep+1)
        eval_s = ba.eval_prediction('tmp2_subtilis', bs_e, bs_t, bs_genes, bs_tfs,'R2')
        eval_a = ba.eval_prediction('tmp2_anthracis', ba_e, ba_t, ba_genes, ba_tfs,'R2')

        (net, bs_genes, bs_tfs) = ba.load_network('tmp2_subtilis')
        aupr += ba.eval_network_pr(net, bs_genes, bs_tfs, BS_priors)
        print aupr/(rep+1)
        acc_s += eval_s
        acc_a += eval_a
    return aupr/repeats


def optimize():
    res1 = scipy.optimize.minimize(multi_species_benchmarks_opt1, [0.5, 0.5], method='Nelder-Mead', options={'maxiter':40})
    res2 = scipy.optimize.minimize(multi_species_benchmarks_opt2, [0.5], method='Nelder-Mead', options={'maxiter':40})
    return (res1, res2)

def test_scad():
    subtilis_conds = range(ba.subtilis_conds)
    random.shuffle(subtilis_conds)    
        
    sub_s = subtilis_conds#[0:int(len(subtilis_conds)*0.05)]
                    
    ba.run_both_scad(lamP=1, lamR=2, lamS=0.1, outf='sub_scad_5.0', sub_s = sub_s, sub_i = [], sub_t = [], it=1, k=50, it_s=3)
    
def bench_lam(s, lamS):
    
    subtilis_conds = range(ba.subtilis_conds)
    random.shuffle(subtilis_conds)    
        
    sub_s = subtilis_conds[0:int(len(subtilis_conds)*0.025)]
                    
    
    ba.run_both(lamP=1, lamR=2, lamS=0, outf=s+'ft_0', sub_s = sub_s, sub_i = [], sub_t = [])

    ba.run_both(lamP=1, lamR=2, lamS=lamS, outf=s+'ft_s', sub_s = sub_s, sub_i = [], sub_t = [])


def rescale_points_cov(c, points):
    (s, v, d) = np.linalg.svd(c)
    print (s, v, d)
    points = np.dot(s, np.dot(np.diag(v), points))
    return points

def make_cov_ellipses(r1, r2, s):
    points = np.zeros((2, 100))
    thetas = np.linspace(0, 2*np.pi, 100)
    for theta_i in range(100):
        theta = thetas[theta_i]
        points[0, theta_i] = np.cos(theta)
        points[1, theta_i] = np.sin(theta)
    c0 = make_cov(r1, r2, 0, True)
    c1 = make_cov(r1, r2, s, True)
    c2 = make_cov(r1, r2, s, False)
    points0 = rescale_points_cov(c0, points)
    points1 = rescale_points_cov(c1, points)
    points2 = rescale_points_cov(c2, points)
    from matplotlib import pyplot as plt
    plt.plot(points0[0,:], points0[1,:], 'r')
    plt.hold(True)
    plt.plot(points2[0,:], points2[1,:], 'g')
    plt.plot(points1[0,:], points1[1,:], 'b')
     

    plt.legend(['no fusion', 'unadjusted ridge', 'adjusted ridge'])
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xlabel('coefficient 1')
    plt.ylabel('coefficient 2')
    plt.show()

    
def make_invcov(R1, R2, S, adj):
    if adj:
        (R1, R2) = fl.adjust(R1, R2, S)
    einv1 = np.zeros((2,2))
    einv2 = np.zeros((2,2))
    einv1[0,0] = S + R1
    einv1[0,1] = -S
    einv1[1,0] = -S
    einv1[1,1] = S+R2
    
    return einv1

def make_cov(r1, r2, s, adj):
    e1 = make_invcov(r1, r2, s, adj)
    return np.linalg.inv(e1)

#make a pretty picture
def draw_scad(lamS, a):
    from matplotlib import pyplot as plt
    b1vals = np.linspace(-10,10,100)
    fuse_constraint = fl.constraint(fl.coefficient(0, 0, 0), fl.coefficient(1, 0, 0), 1)
    fuse_constraints = [fuse_constraint]
    b2 = np.zeros((1,1))
    b1 = np.zeros((1,1))
    penalties = []
    for b1val in b1vals:
        b1[0,0] = b1val
        fc = fl.scad([b1, b2], fuse_constraints, lamS, a)
        penalties.append(fc[0].lam)

    pv = np.cumsum(penalties*b1vals)
    
    plt.plot(b1vals, pv - np.min(pv))
    plt.hold(True)
    plt.plot([a, a], [0, np.max(pv-np.min(pv))], '--k')
    plt.plot([-a, -a], [0, np.max(pv-np.min(pv))], '--k')

    plt.plot([lamS, lamS], [0, np.max(pv-np.min(pv))], 'k')
    plt.plot([-lamS, -lamS], [0, np.max(pv-np.min(pv))], 'k')
    #plt.ylim(0,12)
    plt.xlabel('|B0 - B1|')
    plt.ylabel('saturating penalty')
    #plt.figure()
    #plt.plot(b1vals, penalties)
    #plt.ylim(-0.1,1.1)
    #plt.xlabel('|B0 - B1|')
    #plt.ylabel('L2 fusion penalty weight')
    #plt.show()
    
#multi_species_benchmarks2()
