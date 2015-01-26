import numpy as np
import fused_reg as fr
import random
import os

#SECTION: -------------UTILITY FUNCTIONS------------------------------
#quantile normalizes conditions AND scales to mean zero/unit variance
#AND DOES NOT divides expression matrices by the square root of the sample size
#CHANGED! mean subtract after quantile normalizing
def normalize(exp_a, mean_zero = False):
        
    canonical_dist = np.sort(exp_a, axis=1).mean(axis=0)
    #if mean_zero:
    #    canonical_dist = canonical_dist - canonical_dist.mean()
    
    canonical_dist = canonical_dist / canonical_dist.std()
    
    exp_n_a = np.zeros(exp_a.shape)
    for r in range(exp_a.shape[0]):
        order = np.argsort(exp_a[r, :])
        exp_n_a[r, order] = canonical_dist
    #exp_n_a / np.sqrt(exp_n_a.shape[0])
    if mean_zero:
        exp_n_a = exp_n_a - exp_n_a.mean(axis=0)

    return exp_n_a


#creates a new expression matrix by joining two exp mats, with arbitrary gene coverage
#keeps only the genes that are shared in common
def join_expr_data(names1, names2, exp_a1, exp_a2):
    names = list(set(names1).intersection(set(names2)))
    name_to_ind = {names[x] : x for x in range(len(names))}
    
    def n_to_i(n):
        if n in name_to_ind:
            return name_to_ind[n]
        return -1
    
    exp_a = np.zeros((exp_a1.shape[0] + exp_a2.shape[0], len(names)))
    
    i=0
    #name_ind_to_name1_ind[i] maps an index in names to an index in names1
    name_ind_to_name1_ind = {n_to_i(names1[x]) : x for x in range(len(names1))}

    #name1_fr_inds[i] is the index in names1 that corresponds ti names[i]
    name1_fr_inds = map(lambda x: name_ind_to_name1_ind[x], range(len(names)))
    #array of the previous
    name1_fr_inds_a = np.array(name1_fr_inds)
    #copy into the array
    for r1 in range(exp_a1.shape[0]):
        exp_a[r1, :] = exp_a1[r1, name1_fr_inds_a]
    name_ind_to_name2_ind = {n_to_i(names2[x]) : x for x in range(len(names2))}
    name2_fr_inds = map(lambda x: name_ind_to_name2_ind[x], range(len(names)))
    name2_fr_inds_a = np.array(name2_fr_inds)
    for r2 in range(exp_a2.shape[0]):
        exp_a[r2+exp_a1.shape[0], :] = exp_a2[r2, name2_fr_inds_a]

    return (exp_a, names)


#SECTION: -----------------------------DATA LOADERS------------------

class data_source():
    def __init__(self):
        print 'NOT IMPLEMENTED!'
        self.N = 0
        self.name = 'cookie monster junk DNA'

    #returns a list of k (approximately) equally sized random folds.
    #folds are lists of integers referring to condition numbers 
    def partition_data(self,k):
        conds = np.arange(self.N)
        random.shuffle(conds)
        incr = float(self.N)/k
        upper_bounds = np.arange(k+1) * self.N / float(k)
        partitions = []
        for i in range(k):
            low = int(upper_bounds[i])
            high = int(upper_bounds[i+1])
            partitions.append(conds[low:high])
        return partitions
    
    #loads data associated with a list of conditions
    def load_data(self,conditions=None):
        if conditions == None:
            conditions = np.arange(self.N)
        conditions = np.array(conditions)
        exp_mat = self.exp_mat[conditions, :]
        tf_mat = self.tf_mat[conditions, :]
        return (exp_mat, tf_mat, self.genes, self.tfs)
    
    #returns a list of priors associated with this data source
    def get_priors(self):
        print 'NOT IMPLEMENTED!'
        return None

#this class assumes directory layout we use for generated data
class standard_source(data_source):
    def __init__(self, datadir, org_ind):
        #strip last separator
        if datadir[-1] == os.sep:
            datadir = datadir[0:-1]
        if org_ind == 0:
            expr_f = file(datadir + os.sep + 'expression1')
            prior_fn = os.path.join(datadir,'priors1')
            tfs_f = file(datadir + os.sep + 'tfnames1')
            name_key = 'organism1'
        else:
            expr_f = file(datadir + os.sep + 'expression2')
            prior_fn = os.path.join(datadir,'priors2')
            tfs_f = file(datadir + os.sep + 'tfnames2')
            name_key = 'organism2'
        #first get the name by scanning the description
        descr_f = file(datadir + os.sep + 'description')
        fslt = map(lambda x: x.split('\t'), descr_f.read().split('\n'))
        name = 'dunno'
        for entry in fslt:
            if len(entry) == 2:
                if entry[0] == name_key:
                    name = entry[1]
        descr_f.close()
        #now get the expression data
        expr_fs = expr_f.read()
        expr_f.close()
        expr_fsn = filter(len, expr_fs.split('\n'))
        expr_fsnt = map(lambda x: x.split('\t'), expr_fsn)
        
        genes = expr_fsnt[0]
        nconds = len(expr_fsnt)-1
        
        exp_mat = np.zeros((nconds, len(genes)))
        for row in range(nconds):
            expr = expr_fsnt[row + 1]
            expr_arr = np.array(expr)
            exp_mat[row,:] = expr_arr
        #now get the tf names
        tfs_fs = tfs_f.read()
        tfs_f.close()
        tfs_fsn = filter(len, tfs_fs.split('\n'))
        
        tfs = tfs_fsn
        #now pull out that part of the expression data as the TF expr matrix
        
        tfs_set = set(tfs)
        gene_is_tf = np.array(map(lambda x: x in tfs_set, genes))
        
        tf_mat = exp_mat[:, gene_is_tf]

            
        self.datadir = datadir
        self.exp_mat = exp_mat
        self.tf_mat = tf_mat
        self.genes = genes
        self.tfs = tfs
        self.N = exp_mat.shape[0]
        self.prior_fn = prior_fn
        self.name = name
        
        
    def get_priors(self):
        p = file(self.prior_fn)
        ps = p.read()
        psn = filter(len, ps.split('\n'))
        psnt = map(lambda x: x.split('\t'), psn)
        priors = map(lambda x: (fr.one_gene(x[0], self.name), fr.one_gene(x[1], self.name)), psnt)
        
        signs = []
        for x in psnt:
            sign = x[2]
            if sign == 'activation':
                signs.append(1)
            if sign == 'repression':
                signs.append(-1)
            else:
                signs.append(0)

        p.close()
        
        return (priors, signs)

#SUBSECTION: ----------------------REAL DATA----------------------



class ba_timeseries(data_source):
    def __init__(self):
        f = file('data/bacteria1/Normalized_data_RMA._txt')
        fs = f.read()
        fsn = filter(len, fs.split('\n'))
        fsnt = map(lambda x: x.split('\t'), fsn)
        conds = fsnt[0][1:]
    #first line is SCAN REF
    #second line is composite element REF
    #lines 3-end are data
        exp_mat_t = np.zeros((len(fsnt)-2, len(conds)))#first col gene
        genes = []
        f_tf = file('data/bacteria1/tfNamesAnthracis')
        f_tfs = f_tf.read()
        tfs = filter(len, f_tfs.split('\n'))


        for r in range(exp_mat_t.shape[0]):
            gene_str_full = fsnt[r+2][0]
        #gene name is 4th element, separated by ':'
            gene_str = gene_str_full.split(':')[3]
            gene_str = gene_str.replace('_pXO1_','').replace('_pXO2','')#what is this? dunno!
            expr = np.array(fsnt[r+2][1:])
            exp_mat_t[r, :] = expr
            genes.append(gene_str)

    #require that tfs be genes that we have data for!
        tfs = filter(lambda x: x in genes, tfs)

        tf_mat_t = np.zeros((len(tfs), len(conds)))

        gene_to_ind = {genes[x] : x for x in range(len(genes))}
        for ti in range(len(tfs)):
            gi = gene_to_ind[tfs[ti]]
            tf_mat_t[ti, :] = exp_mat_t[gi, :]
        exp_mat = exp_mat_t.T
    
        tf_mat = tf_mat_t.T

        self.exp_mat = exp_mat
        self.tf_mat = tf_mat
        self.genes = genes
        self.tfs = tfs
        self.N = exp_mat.shape[0]
        
        self.name = 'ba_iron'

    #returns a list of priors associated with this data source
    def get_priors(self):
        return ([],[])
        


#B anthracis data relating to iron starvation conditions
class ba_iron(data_source):
    def __init__(self):
        f = file('data/bacteria1/normalizedgeneexpressionvalues.txt')
        fs = f.read()
        fsn = filter(len, fs.split('\n'))
        fsnt = map(lambda x: x.split('\t'), fsn)
        conds = fsnt[0][1:]
    #first line is SCAN REF
    #second line is composite element REF
    #lines 3-end are data
        f_tf = file('data/bacteria1/tfNamesAnthracis')
        f_tfs = f_tf.read()
        tfs = filter(len, f_tfs.split('\n'))
        
        exp_mat_t = np.zeros((len(fsnt)-2, len(conds)))#first col gene
        genes = []

        for r in range(exp_mat_t.shape[0]):
            gene_str = fsnt[r+2][0]
        
        #gene name is 4th element, separated by ':'
        
            expr = np.array(fsnt[r+2][1:])
            exp_mat_t[r, :] = expr
            genes.append(gene_str)

    #require that tfs be genes that we have data for!
        tfs = filter(lambda x: x in genes, tfs)

        tf_mat_t = np.zeros((len(tfs), len(conds)))

        gene_to_ind = {genes[x] : x for x in range(len(genes))}
        for ti in range(len(tfs)):
            gi = gene_to_ind[tfs[ti]]
            tf_mat_t[ti, :] = exp_mat_t[gi, :]
        exp_mat = exp_mat_t.T

        tf_mat = tf_mat_t.T
        self.exp_mat = normalize(exp_mat, True)
        self.tf_mat = normalize(tf_mat, False)
        self.genes = genes
        self.tfs = tfs
        self.N = exp_mat.shape[0]
        
        self.name = 'ba_time'
             
    #returns a list of priors associated with this data source
    def get_priors(self):
        return ([],[])

class subt(data_source):
    def __init__(self):
    
        expr_fn = 'data/bacteria1/B_subtilis.csv'
        tfs_fn = 'data/bacteria1/tfNames_subtilis.txt'
        f = file(expr_fn)
        fs = f.read()
        fsl = filter(len, fs.split('\n'))
        fslc = map(lambda x: x.split(','), fsl)
        f.close()
    
        t = file(tfs_fn)
        ts = t.read()
        tfs = filter(len, ts.split('\n'))
        t.close()

        tfs_set = set(tfs)


        conds = fslc[0]
        genes = map(lambda x: x[0], fslc[1:])
        exp_mat_t = np.zeros((len(genes), len(conds)))
        for r in range(len(genes)):
            conds_f = map(float, fslc[1+r][1:])
            conds_a = np.array(conds_f)
            exp_mat_t[r, :] = conds_a

        tf_mat_t = np.zeros((len(tfs), len(conds)))
        gene_to_ind = {genes[x] : x for x in range(len(genes))}
    
        for ti in range(len(tfs)):
            gi = gene_to_ind[tfs[ti]]
            tf_mat_t[ti, :] = exp_mat_t[gi, :]

        exp_mat = exp_mat_t.T
        tf_mat = tf_mat_t.T

        self.exp_mat = normalize(exp_mat, True)
        self.tf_mat = normalize(tf_mat, False)
        self.genes = genes
        self.tfs = tfs
        self.N = exp_mat.shape[0]
        self.offset = 0
        self.name = 'B_subtilis'

    def get_priors(self):
        priors_fn = 'data/bacteria1/gsSDnamesWithActivitySign082213'
        p = file(priors_fn)
        ps = p.read()
        psn = filter(len, ps.split('\n'))
        psnt = map(lambda x: x.split('\t'), psn)
        priors = map(lambda x: (fr.one_gene(x[0], self.name), fr.one_gene(x[1], self.name)), psnt)
        signs = map(lambda x: [-1,1][x[2]=='activation'], psnt)
        p.close()
        return (priors, signs)


#NOTE: I just changed anthracis to quantile normalize separately before combining. I can't see a downside to doing it this way, or the other way
class anthr(data_source):
    def __init__(self):
        self.ba_i = ba_iron()
        self.ba_t = ba_timeseries()
        self.name = 'B_anthracis'
        (e, genes) = join_expr_data(self.ba_i.genes, self.ba_t.genes, self.ba_i.exp_mat, self.ba_t.exp_mat)
        (t, tfs) = join_expr_data(self.ba_i.tfs, self.ba_t.tfs, self.ba_i.tf_mat, self.ba_t.tf_mat)
        genes = map(lambda x: x.replace('_at',''), genes)
        tfs = map(lambda x: x.replace('_at',''), tfs)


        self.exp_mat = e
        self.genes = genes
        self.tf_mat = t
        self.tfs = tfs
        self.N = self.exp_mat.shape[0]

    #returns a list of k (approximately) equally sized random folds.
    #folds are lists of integers referring to condition numbers 
    def partition_data(self, k):
        par1 = self.ba_i.partition_data(k)
        par2 = self.ba_t.partition_data(k)
        partitions = []
        off = self.ba_i.N #add this offset to the timeseries conds
        for i in range(k):
            partitions.append(np.hstack((par1[i], off + par2[i])))
        return partitions
    
    def get_priors(self):
        return ([], [])

#NOTE: I am missing the (largely unsuccessful) loader for subtilis that takes into account timeseries data



#SECTION: ----------------------------------ORTHOLOGY LOADERS----------

def load_orth(orth_fn, organisms):
    f = file(orth_fn)
    fs = f.read()
    fsn = filter(len, fs.split('\n'))
    fsnt = map(lambda x: x.split('\t'), fsn)
    
    orths = []
    for o in fsnt:
    
        orths.append([fr.one_gene(name=o[0],organism=organisms[0]), fr.one_gene(name=o[1], organism=organisms[1])])
    return orths

ba_bs_orth = lambda: load_orth('data/bacteria1/bs_ba_ortho_804',['B_anthracis','B_subtilis'])


#SECTION: ----------------------------DATA GENERATORS------------------
#generates a matrix Y from the linear model specified by B.
#x is sampled uniformly from -1 to 1
#input N: the number of samples
#input B: matrix specifying linear model
#input noise_std: std of noise!
#returns: (X, Y)
def generate_from_linear(N, B, noise_std):
    X = 1-2*np.random.random((N, B.shape[0]))
    Y = np.dot(X,B) + noise_std*np.random.randn(N, B.shape[1])
    return (X, Y)    

#This function just corrects the strange decision i made to not use one_gene objects. I don't think there's anything more to it than this?
#NOTE: this function appeared to throw away all of the mappings but the first... I've corrected this here
def omap_to_orths(omap):
    orths = []
    for gene1 in omap.keys():       
        for gene2 in omap[gene1]:
            orths.append( (gene1, gene2) )
    return orths

#this functions somehow generates orthology mappings, and adds them to omap
#this strange design was chosen over simply returning an orthology mapping because i am a sadist
#this works by generating orthology groups of random size from 2 to max_grp_size (inclusive). Each organism must contribute at least one gene to the orthology group (this could change later). This process continues until the total number of genes in orthology groups is approximately pct_fused percent of the total genes. 
#what goes into the orthology mapping! well.... it seems to be a dictionary mapping organism name/gene tuples to lists of organism name/gene tuples. This could be rewritten to use one_gene objects, because that's what they're for, but really the whole thing should be burned to the ground
def build_orth(genes1, genes2, max_grp_size, pct_fused, omap, organisms, shuffle=True):
    
    if shuffle:
        random.shuffle(genes1)
        random.shuffle(genes2)
    
    amt_fused = np.round((len(genes1)+len(genes2))*pct_fused)
    ind1 = 0
    ind2 = 0
    while ind1 + ind2 < amt_fused:
        grp_size = random.randrange(2, max_grp_size+1)

        grp1_size = random.randrange(1, grp_size)
        grp2_size = grp_size - grp1_size
        #modify the group sizes to deal with not enough of one sub
        grp1_size = min(grp1_size, len(genes1)-ind1)
        grp2_size = min(grp2_size, len(genes2)-ind2) 
        
        
        #enumerate ALL of the 1-1 orthologies that span different organisms within this group
        #fixed a bug here wherein if the two species shared gene names, things fucked up. this can be fixed by only considering one direction of orthology.
        for i in range(ind1, ind1 + grp1_size):
            for j in range(ind2, ind2 + grp2_size):
                
                sub1c = fr.one_gene(genes1[i], organism = organisms[0])
                sub2c = fr.one_gene(genes2[j], organism = organisms[1])
                
                if sub1c in omap:
                    omap[sub1c].append(sub2c)
                else:
                    omap[sub1c] = [sub2c]
        ind1 += grp1_size
        ind2 += grp2_size
    

#TO THE BEST OF MY KNOWLEDGE THIS IS WHAT HAPPENS. this function builds two bata matrices, with dims specified by b1/tfg_count2. pct_fused percent of genes and TFs are (separately) assigned to orthology groups of size 2-max_grp_size (sampled uniformly). Coefficientsa are assigned value from standard normal distributions. Coefficients that are fused (that is, both gene and tf are fused) are assigned to the same value, perturbed by fuse_std. Sparse specifies the proportion of fused groups of coefficients (including groups of size 1) that are set to 0
#NOTE: tfg_count1 is the number of tfs and non-tf genes in a tuple
def fuse_bs_orth(tfg_count1, tfg_count2, max_grp_size, pct_fused, fuse_std, sparse, organisms):
    #create empty expression matrices
    b1 = np.nan * np.ones(tfg_count1)
    b2 = np.nan * np.ones(tfg_count2)
    tfs1 = map(lambda x: organisms[0]+'t'+str(x), range(b1.shape[0]))
    tfs2 = map(lambda x: organisms[1]+'t'+str(x), range(b2.shape[0]))   
    #THESE ARE GENES THAT ARE NOT ALSO TFS
    #This is done so that we can generate orthologies between genes and tfs separately
    genes1 = map(lambda x: organisms[0]+'g'+str(x), range(b1.shape[0], b1.shape[0]+b1.shape[1]))
    genes2 = map(lambda x: organisms[1]+'g'+str(x), range(b2.shape[0], b2.shape[0]+b2.shape[1]))
    omap = dict()
    build_orth(tfs1, tfs2, max_grp_size, pct_fused, omap, organisms,shuffle=False)
    build_orth(genes1, genes2, max_grp_size, pct_fused, omap, organisms,shuffle=False)
    orths = omap_to_orths(omap)
    bs = [b1, b2]
    genes = [genes1, genes2]
    tfs = [tfs1, tfs2]
    gene_inds = [{genes1[i] : i for i in range(len(genes1))}, {genes2[i] : i for i in range(len(genes2))}]
    tf_inds = [{tfs1[i] : i for i in range(len(tfs1))}, {tfs2[i] : i for i in range(len(tfs2))}]

    orths_query = set(map(tuple, orths))
    
    #recursively fills everything connected to r,c,organism
    #NOTE: we are relying on the fact that transcription factors are also genes
    def fill(r, c, val, std, organism):
        
        fill_val = val + np.random.randn()*std
        organism_ind = organisms.index(organism) #blegh
        if not np.isnan(bs[organism_ind][r,c]): #already filled, return
            return           
        bs[organism_ind][r,c] = fill_val
        
        for orth1 in orths:
            #find an ortholog for row and an ortholog for column
            #do this in both directions - i don't want to require that orthologies be listed both ways
            for rep in [0,1]:
                if tfs[organism_ind][r] == orth1[0].name and organism == orth1[0].organism:
                    for orth2 in orths:
                        if genes[organism_ind][c] == orth2[0].name and organism == orth2[0].organism:
                            organism_r_orth_ind = organisms.index(orth1[1].organism)
                            organism_c_orth_ind = organisms.index(orth2[1].organism)
                            if organism_r_orth_ind == organism_c_orth_ind:
                                r_orth = tf_inds[organism_r_orth_ind][orth1[1].name]
                                c_orth = gene_inds[organism_c_orth_ind][orth2[1].name]
                                fill(r_orth, c_orth, val, std, organisms[organism_r_orth_ind])
                orth1 = (orth1[1], orth1[0]) #repeat in reverse
    #go through each value in each beta matrix and, if not yet filled, fill it in along with all of its linked values
    for organism_ind, b in enumerate(bs):
        for r in range(b.shape[0]):
            for c in range(b.shape[1]):
                if np.isnan(b[r,c]):
                    if np.random.random() < sparse:
                        val = 0
                        std = 0
                    else:
                        val = np.random.randn()
                        std = fuse_std
                    fill(r, c, val, std, organisms[organism_ind])

    return (b1, b2, orths, genes1, tfs1, genes2, tfs2)


# generates priors from a given beta matrix B, some of which may be wrong or missing
def generate_faulty_priors(B, genes, tfs, falsepos, falseneg):
    priors = []
    fakepriors = []
    for r in range(B.shape[0]):
        for c in range(B.shape[1]):
            if B[r, c] != 0:
                priors.append((tfs[r], genes[c]))
            if B[r, c] == 0:
                fakepriors.append((tfs[r], genes[c]))
    num_to_remove = int(falseneg * len(priors))
    num_to_add = int(falsepos*(len(priors) - num_to_remove)/(1-falsepos))    
    random.shuffle(priors)
    random.shuffle(fakepriors)
    return priors[0:(len(priors) - num_to_remove)] + fakepriors[0:num_to_add]


def generate_faulty_orth(orths, genes1, tfs1, genes2, tfs2, organisms, falsepos, falseneg):
    #make a list of sets containing gene fusion groups to prevent from adding false orths that result in unduly large fusion groups

    num_to_remove = int(falseneg * len(orths))
    num_to_add = int(falsepos*(len(orths) - num_to_remove)/(1-falsepos))
    #print falsepos
    #print num_to_add
    # f = add / (base + add)
    # f*base + f*add = add
    # f*base = (1+f)*add
    num_to_add = 5
    orth_genes = set()
    for orth in orths:
        orth_genes.add(orth[0])
        orth_genes.add(orth[1])
    
    all_possible_orths = []
    for gene1 in genes1:
        for gene2 in genes2:
            all_possible_orths.append((fr.one_gene(name=gene1, organism = organisms[0]), fr.one_gene(name=gene2, organism = organisms[1])))
    for tf1 in genes1:
        for tf2 in genes2:
            all_possible_orths.append((fr.one_gene(name=tf1, organism = organisms[0]), fr.one_gene(name=tf2, organism = organisms[1])))
    random.shuffle(all_possible_orths)
    random.shuffle(orths)

    to_add = []
    for candidate_orth in all_possible_orths:
        print candidate_orth
        print orth_genes
        if len(to_add) == num_to_add:
            break
        if candidate_orth[0] in orth_genes or candidate_orth[1] in orth_genes:
            continue
        to_add.append(candidate_orth)
        

    return orths[0:(len(orths) - num_to_remove)] + to_add


    



#writes fake data, assumes some reasonable defaults
def write_fake_data1(out_dir=None, tfg_count1=(5,10), tfg_count2=(5,10), N1=10, N2=10, max_grp_size=2, pct_fused=1.0, fuse_std=0.5, sparse=0.0, organisms = ['uno','dos'], prior_falsepos=0.0, prior_falseneg=0.0, orth_falsepos=0.0, orth_falseneg=0.0, measure_noise1=0.1, measure_noise2=0.1):

    (B1, B2, orths, genes1, tfs1, genes2, tfs2) = fuse_bs_orth(tfg_count1, tfg_count2, max_grp_size, pct_fused, fuse_std, sparse, organisms)


    
    (x1, y1) = generate_from_linear(N1, B1, measure_noise1)
    (x2, y2) = generate_from_linear(N2, B2, measure_noise2)
    orths = generate_faulty_orth(orths, genes1, tfs1, genes2, tfs2, organisms, orth_falsepos, orth_falseneg)
    
    (genes1c, expr1) = concat_tfs_genes(genes1, tfs1, x1, y1)
    (genes2c, expr2) = concat_tfs_genes(genes2, tfs2, x2, y2)
    #(genes1c, expr1) = (genes1, x1)
    #(genes2c, expr2) = (genes2, x2)

    priors1 = generate_faulty_priors(B1, genes1c, tfs1, prior_falsepos, prior_falseneg)
    priors2 = generate_faulty_priors(B2, genes2c, tfs2, prior_falsepos, prior_falseneg)

    
    if out_dir == None:
        out_dir = os.tempnam('data/fake_data','gdat')
        print 'putting data in ' + out_dir
    #now we do writing to a file stuff
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    
    write_expr_mat(out_dir+os.sep+'expression1', expr1, genes1c)
    write_expr_mat(out_dir+os.sep+'expression2', expr2, genes2c)
    write_priors(out_dir+os.sep+'priors1',priors1)
    write_priors(out_dir+os.sep+'priors2',priors2)
    write_tfnames(out_dir+os.sep+'tfnames1',tfs1)
    write_tfnames(out_dir+os.sep+'tfnames2',tfs2)
    #since this is simulated data the first part of B is an identity matrix. That is, the TF expressions appear out of nowhere.
    #does this make me happy? no it does not.
    #maybe someday TFs can be some left-eigenvector with eigenvalue 1 of an interesting B, but not today
    write_network(genes1c, tfs1, np.hstack((np.eye(B1.shape[0]), B1)), os.path.join(out_dir, 'beta1'))
    write_network(genes2c, tfs2, np.hstack((np.eye(B2.shape[0]), B2)), os.path.join(out_dir, 'beta2'))



    write_orth(out_dir+os.sep+'orth', orths)
    f = file(out_dir+os.sep+'description', 'w')
    f.write('tfg_count1: %s\ntfg_count2\t%s\nN1\t%d\nN2\t%d\nmax_grp_size\t%d\npct_fused\t%f\nfuse_std:%f\nsparse\t%f\norganism1\t%s\norganism2\t%s\nprior_falsepos\t%f\nprior_falseneg\t%f\nmeasure_noise1\t%f\nmeasure_noise2\t%f\n' % (str(tfg_count1), str(tfg_count2), N1, N2, max_grp_size, pct_fused, fuse_std, sparse, organisms[0], organisms[1], prior_falsepos, prior_falseneg, measure_noise1, measure_noise2))
    f.close()
    return out_dir

#adds tf expressions to the gene expression matrix, and tfs to the list of genes. This better matches the real data, in which tfs are genes.
#returns new genes, and gene expression mat
#NOTE: this requires that genes and tfs have disjoint names.
def concat_tfs_genes(genes, tfs, x, y):
    
    expr_mat = np.hstack((y, x))
    #check if they have disjoint names
    if len(set(genes).intersection(set(tfs))):
        print 'WAKA WAKA WAKA ALERT YOU HAVE GENES AND TFS WITH THE SAME NAMES'
        raise Warning #what does this do?
    genes = tfs + genes
    return (genes, expr_mat)

#SECTION: code for writing things that may or may not be fake

#write an expression matrix
#format is: line 1, gene names. line 2-(N+1) expressions.
#tab delimited
def write_expr_mat(outf, expr, genes):
    f = file(outf,'w')
    head = '\t'.join(genes) + '\n' #os independent linebreaks? fuck that.
    f.write(head)
    
    for cond in range(expr.shape[0]):
        gene_expressions = map(str, expr[cond, :])
        f.write('\t'.join(gene_expressions) + '\n')
    f.close()

#writes priors
#format is: lines 1-N: tf gene sign
#signs are 'activation' 'repression' 'dunno'
#priors as input is a list of tuples containing gene/tf names
def write_priors(outf, priors, signs=None):
    f = file(outf, 'w')
    for i, prior in enumerate(priors):
        (tf, gene) = prior
        if signs == None:
            sign = 'dunno'
        else:
            sign = signs[i]
        f.write('%s\t%s\t%s\n' % (tf, gene, sign))
        #f.write('%s\t%s\t%s\n' % (tf.name, gene.name, sign))
    f.close()

#write orth.
#format is: lines 1-N gene gene
#orth is pairs of one_genes
def write_orth(outf, orth):
    f = file(outf, 'w')
    for o in orth:
        gene1 = o[0].name
        gene2 = o[1].name
        f.write('%s\t%s\n' % (gene1, gene2))

    f.close()

#writes tf names separated by newlines
def write_tfnames(outf, tfnames):
    f = file(outf, 'w')
    f.write('\n'.join(tfnames)+'\n')
    f.close()

#writes a network
def write_network(genes, tfs, B, outf):
    
    Bs_str_l = []
    Bs_str_l.append('\t'.join(tfs))
    for gi in range(len(genes)):
        gene = genes[gi]
        regulators = B[:, gi]
        
        Bs_str_l.append(gene +'\t'+ '\t'.join(map(str, regulators)))
    f = file(outf, 'w')
    f.write('\n'.join(Bs_str_l))
    f.close()

#loads a network
def load_network(net_fn):
    f = file(net_fn)
    fs = f.read()
    fsl = filter(len, fs.split('\n'))
    fslt = map(lambda x: x.split('\t'), fsl)
    tfs = fslt[0]
    genes = map(lambda x: x[0], fslt[1:])
    #the network is written as the transpose of the matrix we want
    net = np.zeros((len(genes), len(tfs)))
    for g in range(len(genes)):
        targets = np.array(map(float, fslt[g+1][1:]))
        net[g, :] = targets
    return (net.T, genes, tfs)

#writes real data to a standard formatted folder. ONLY RUN ONCE!
#if you need to run again, change f.write('%s\t%s\t%s\n' % (tf, gene, sign)) to f.write('%s\t%s\t%s\n' % (tf.name, gene.name, sign))

def voodoo():
    sub = subt()
    anth = anthr()
    
    (bs_e, bs_t, bs_genes, bs_tfs) = sub.load_data()
    (ba_e, ba_t, ba_genes, ba_tfs) = anth.load_data()

    (bs_priors, bs_sign) = sub.get_priors()
    (ba_priors, ba_sign) = anth.get_priors()
    orths = ba_bs_orth()
    
    out_dir = os.path.join('data','bacteria_standard')
    os.mkdir(out_dir)
    write_expr_mat(out_dir+os.sep+'expression1', bs_e, bs_genes)
    write_expr_mat(out_dir+os.sep+'expression2', ba_e, ba_genes)
    write_priors(out_dir+os.sep+'priors1',bs_priors)
    write_priors(out_dir+os.sep+'priors2',ba_priors)
    write_tfnames(out_dir+os.sep+'tfnames1',bs_tfs)
    write_tfnames(out_dir+os.sep+'tfnames2',ba_tfs)
    write_orth(out_dir+os.sep+'orth', orths)
    f = file(out_dir+os.sep+'description', 'w')
    f.write('organism1\t%s\norganism2\t%s' % ('B_subtilis','B_anthracis'))
    f.close()

    
