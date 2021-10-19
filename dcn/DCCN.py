import numpy as np
import sys

from dcn.hmm import HMM
from dcn.CNET import CNET
from dcn.ccnet import CNET_CLT

usage_str = """Usage:
======
python DCCN.py <dir-path> <file-name> <prior-depth> <trans-depth> <function> <alpha> [ <evid-idx> <K> ]

Arguments:
==========
<dir-path>:\tPath to directory that contains the datasets
<file-name>:\tName of the dataset
<prior-depth>:\tMaximum depth of prior cutset network
<trans-depth>:\tMaximum depth of transition conditional cutset network (CCN)
<function>:\tFunction to be used for CCN (LR or NN)
<alpha>:\tHyperparameter for neural network
<evid-idx>:\tEvidence indices for computing CLL (optional)
<K>:\tNumber of particles for particle filtering (optional; default: 10)"""

class DCCN:
    def __init__(self):
        self.prior_model = None
        self.transition_model = None
        self.size = 0
    def learn(self, dset, seqs, max_depth_prior=0, max_depth=0, function="LR", alpha=0.01):
        # Validation
        if (dset.ndim!=2 or dset.shape[0]==0 or seqs.ndim!=1 or dset.shape[0] != seqs.sum() or \
                function not in ["LR", "NN"] or alpha<0.0 ):
            return
        # Initialize models
        self.prior_model = CNET(depth=max_depth_prior)
        self.transition_model = CNET_CLT([], max_depth)
        self.size = dset.shape[1]
        # Learn parameters
        print("** Learning prior model")
        self.prior_model.learnStructure(train_dset)
        print("** Learning transition model")
        transition_dset = HMM.create_transition_dset(dset, seqs)
        evid_var = np.arange(train_dset.shape[1])
        query_var = np.arange(train_dset.shape[1], 2 * train_dset.shape[1])
        evid_arr = transition_dset[:, evid_var]
        query_arr = transition_dset[:, query_var]
        self.transition_model.learnStructure(evid_arr, query_arr, function, alpha)
    def avgLL(self, dset, seqs, evid_idx=np.array([]), K=100, L=1):
        np.random.seed(1)
        # Validation
        if (dset.ndim!=2 or dset.shape[0]==0 or seqs.ndim!=1 or dset.shape[0] != seqs.sum()):
            return 1.0
        # Initialize Variables
        total_ll = 0.0
        total_ell = 0.0
        cum_test_seqs = np.append(0, test_seqs[:-1]).cumsum()
        bool_evid_idx = np.in1d(np.arange(test_dset.shape[1]), evid_idx)
        # Process each sequence
        for s in range(test_seqs.shape[0]):
            # First compute likelihood of Pr(x,e)
            local_ll = self.prior_model.computeLL(test_dset[cum_test_seqs[s]:cum_test_seqs[s] + 1])
            if (local_ll>0.0): return 1.0
            total_ll += local_ll
            # Next, compute likelihood of Pr(e)
            if (evid_idx.size):
                prev_particles = self.prior_model.weighted_samples(K, bool_evid_idx,
                                                                    test_dset[cum_test_seqs[s]][evid_idx])
                local_ell = np.log(self.prior_model.prob_evid(self.prior_model.tree, bool_evid_idx,
                                                              test_dset[cum_test_seqs[s]][evid_idx]))
                if (local_ell > 0.0): return 1.0
                total_ell += local_ell
            # Repeat above steps for all other members of sequence
            for i in range(cum_test_seqs[s] + 1, cum_test_seqs[s] + test_seqs[s]):
                local_ll = self.transition_model.computeLL(test_dset[i - 1:i], test_dset[i:i + 1])
                if (local_ll > 0.0): return 1.0
                total_ll += local_ll
                # If evidence is present, do particle filtering
                if (evid_idx.size):
                    new_particles = []
                    i_prob_evid = 0.0
                    W = 0.0
                    # Compute P(e^t,e^{t-1}) by summing out P(e^t,e^{t-1},x^{t-1})
                    for particle in prev_particles:
                        weight, vals = particle
                        evid = test_dset[i][evid_idx]
                        curr_model = self.transition_model.convert_to_cnet(vals.reshape(1, vals.size))
                        i_prob_evid += weight * curr_model.prob_evid(curr_model.tree, bool_evid_idx, evid)
                        new_particles += curr_model.weighted_samples(L, bool_evid_idx,
                                                                     test_dset[i][evid_idx])
                        W += weight
                    # Compute P(e^t|e^{t-1})
                    local_ell = np.log(i_prob_evid/W)
                    if (local_ell > 0.0): return 1.0
                    total_ell += local_ell
                    # Create prev_particles for next time slice
                    prev_particles = sorted(new_particles, key=lambda x: x[0])[::-1][:K]
        return (total_ll-total_ell)/ test_dset.shape[0]

if __name__=="__main__":
    # Check if arguments are present
    if (len(sys.argv) < 7):
        print(usage_str)
        sys.exit(1)
    # Read in arguments
    dir_name = sys.argv[1]
    file_name = sys.argv[2]
    max_depth_prior = int(sys.argv[3])
    max_depth = int(sys.argv[4])
    function = sys.argv[5].strip() if (sys.argv[5].strip() in ["LR", "NN"]) else "LR"
    alpha = float(sys.argv[6])
    evid_idx = np.fromstring(sys.argv[7], sep=",", dtype=int) if (len(sys.argv) > 7) else np.array([])
    K = int(sys.argv[8]) if (len(sys.argv)>8) else 10
    print("==========================")
    print("DCCN Inference")
    print("==========================")
    print("Directory: " + dir_name)
    print("File Name: " + file_name)
    print("Max Depth (Prior): " + str(max_depth_prior))
    print("Max Depth (Transition): " + str(max_depth))
    print("Function: " + function)
    print("Alpha: " + str(alpha))
    print("Evidence Indices: " + str(evid_idx))
    print("K: " + str(K))
    # Load datasets
    print("** Loading data")
    infile_path = dir_name + '/' + file_name
    train_dset = np.loadtxt(infile_path + ".train", delimiter=',', dtype=int)
    test_dset = np.loadtxt(infile_path + ".test", delimiter=',', dtype=int)
    train_seqs = np.loadtxt(infile_path + ".train.seq", delimiter=',', ndmin=1, dtype=int)
    test_seqs = np.loadtxt(infile_path + ".test.seq", delimiter=',', ndmin=1, dtype=int)
    dom_vals = [ vals for vals in np.loadtxt(infile_path + ".doms", skiprows=1, delimiter=",", dtype=int) ]
    print("** Learning model")
    dccn = DCCN()
    dccn.learn(train_dset, train_seqs, max_depth_prior, max_depth, function, alpha)
    print("** Computing LL")
    print(dccn.avgLL(test_dset, test_seqs, evid_idx, K))
