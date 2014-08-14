import bacteria as ba
import numpy as np
import random


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
    lamPs = np.array([0])#np.logspace(0, 3, 10)
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

def multi_species_benchmarks2():
    repeats = 2
    lamPs = np.array([0])#np.logspace(0, 3, 10)
    lamSs = [0]+list(np.logspace(-1.5, 1.5, 5))
    lamRs = [3]#np.logspace(0, 2, 5)
    iron_conds = range(ba.iron_conds)
    timeseries_conds = range(ba.timeseries_conds)
    subtilis_conds = range(ba.subtilis_conds)
    outf = file('benchmark_results/multi_species_bench3','w')
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
                    sub_i = iron_conds[0:int(len(iron_conds)*0.25)]
                    sub_t = timeseries_conds[0:int(len(timeseries_conds)*0.25)]
                    
                    ba.run_both(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp2', sub_s = sub_s, sub_i = sub_i, sub_t = sub_t)
                    
                    sub_st = subtilis_conds[int(len(subtilis_conds)*0.75):len(subtilis_conds)]
                    sub_it = iron_conds[int(len(iron_conds)*0.75):len(iron_conds)]
                    sub_tt = timeseries_conds[int(len(timeseries_conds)*0.75):len(timeseries_conds)]                    

                    (bs_e, bs_t, bs_genes, bs_tfs) = ba.load_B_subtilis(sub_st)
                    (ba_e, ba_t, ba_genes, ba_tfs) = ba.load_B_anthracis(sub_it, sub_tt)

                    eval_s = ba.eval_prediction('tmp2_subtilis', bs_e, bs_t, bs_genes, bs_tfs,'R2')
                    eval_a = ba.eval_prediction('tmp2_anthracis', ba_e, ba_t, ba_genes, ba_tfs,'R2')
                    acc_s += eval_s
                    acc_a += eval_a
                writestr = 'lamP=%f\tlamS=%f\tlamR=%f\tsubtilis=%f\tanthracis=%f\n' % (lamP, lamS, lamR, acc_s/repeats, acc_a/repeats)
                outf.write(writestr)
                print writestr
                    
    outf.close()


def multi_species_benchmarks2():
    repeats = 2
    lamPs = np.array([0])#np.logspace(0, 3, 10)
    lamSs = [0]+list(np.logspace(-1.5, 1.5, 5))
    lamRs = [3]#np.logspace(0, 2, 5)
    iron_conds = range(ba.iron_conds)
    timeseries_conds = range(ba.timeseries_conds)
    subtilis_conds = range(ba.subtilis_conds)
    outf = file('benchmark_results/multi_species_bench3','w')
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
                    sub_i = iron_conds[0:int(len(iron_conds)*0.25)]
                    sub_t = timeseries_conds[0:int(len(timeseries_conds)*0.25)]
                    
                    ba.run_both(lamP=lamP, lamR=lamR, lamS=lamS, outf='tmp2', sub_s = sub_s, sub_i = sub_i, sub_t = sub_t)
                    
                    sub_st = subtilis_conds[int(len(subtilis_conds)*0.75):len(subtilis_conds)]
                    sub_it = iron_conds[int(len(iron_conds)*0.75):len(iron_conds)]
                    sub_tt = timeseries_conds[int(len(timeseries_conds)*0.75):len(timeseries_conds)]                    

                    (bs_e, bs_t, bs_genes, bs_tfs) = ba.load_B_subtilis(sub_st)
                    (ba_e, ba_t, ba_genes, ba_tfs) = ba.load_B_anthracis(sub_it, sub_tt)

                    eval_s = ba.eval_prediction('tmp2_subtilis', bs_e, bs_t, bs_genes, bs_tfs,'R2')
                    eval_a = ba.eval_prediction('tmp2_anthracis', ba_e, ba_t, ba_genes, ba_tfs,'R2')
                    acc_s += eval_s
                    acc_a += eval_a
                writestr = 'lamP=%f\tlamS=%f\tlamR=%f\tsubtilis=%f\tanthracis=%f\n' % (lamP, lamS, lamR, acc_s/repeats, acc_a/repeats)
                outf.write(writestr)
                print writestr
                    
    outf.close()


def bench_lam():
    
    subtilis_conds = range(ba.subtilis_conds)
    random.shuffle(subtilis_conds)    
        
    sub_s = subtilis_conds[0:int(len(subtilis_conds)*0.05)]
                    
    
    ba.run_both(lamP=1, lamR=5, lamS=0, outf='sub_def_s2_0', sub_s = sub_s, sub_i = [], sub_t = [])
    ba.run_both(lamP=1, lamR=5, lamS=5.0, outf='sub_def_s2_5.0', sub_s = sub_s, sub_i = [], sub_t = [])


#multi_species_benchmarks2()
