import numpy as np
import os
import sys

from dcn.hmm import HMM
from dcn.DCCN import DCCN
from dcn.ccnet import CNET_CLT

usage_str = """Usage:
======
python main.py <data-dir> <dset-name> <prior-depth> <trans-depth> <function> <alpha> [ <K> <L> <output-dir> ]

Arguments:
==========
<data-dir>:\tPath to directory that contains the datasets
<dset-name>:\tName of the dataset
<prior-depth>:\tMaximum depth of prior cutset network
<trans-depth>:\tMaximum depth of transition conditional cutset network (CCN)
<function>:\tFunction to be used for CCN (LR or NN)
<alpha>:\tHyperparameter for neural network
<K>:\t\tNumber of samples to use for particle filtering [ default: 10 ]
<L>:\t\tNumber of ranked explanations [ default: 3 ]
<output-dir>:\tPath to directory that contains the datasets [ default: data directory ]"""

class ExpDCCN(DCCN):
    def __init__(self):
        DCCN.__init__(self)
    def learn(self, ground_dset, evid_dset, seqs, max_depth_prior=0, max_depth=0, function="LR", alpha=0.01):
        # Validation
        if (ground_dset.ndim!=2 or ground_dset.shape[0]==0 or seqs.ndim!=1 or \
            ground_dset.shape[0] != seqs.sum() or evid_dset.ndim!=2 or \
            evid_dset.shape[0]==0 or ground_dset.shape[0] != evid_dset.shape[0] or \
            ground_dset.shape[1] != evid_dset.shape[1] or \
            function not in [ "LR", "NN" ] or alpha<0.0 ):
            return
        # Initialize models
        self.prior_model = CNET_CLT([], max_depth_prior)
        self.transition_model = CNET_CLT([], max_depth)
        self.size = ground_dset.shape[1]
        # Learn prior parameters
        V = self.size
        cum_seqs = np.append(0, seqs.cumsum())
        sensor_dset = np.hstack((ground_dset, evid_dset))
        evid_var = np.arange(V, 2 * V)
        query_var = np.arange(V)
        evid_arr = sensor_dset[:, evid_var]
        query_arr = sensor_dset[:, query_var]
        self.prior_model.learnStructure(evid_arr, query_arr, function, alpha)
        # Learn transition parameters
        keep_idx = np.hstack((np.arange(2 * V, 3 * V), np.arange(V),
                                np.arange(3 * V, 4 * V)))
        transition_dset = HMM.create_transition_dset(sensor_dset, seqs)[:, keep_idx]
        evid_var = np.arange(self.size, 3 * self.size)
        query_var = np.arange(self.size)
        evid_arr = transition_dset[:, evid_var]
        query_arr = transition_dset[:, query_var]
        self.transition_model.learnStructure(evid_arr, query_arr, function, alpha)
    def pfCompile(self, evid_dset, seqs, out_dir='.', dset_name='results', K=10, L=3, seed=1991):
        # Initialization
        np.random.seed(seed)
        V = self.size
        N = len(seqs)
        cum_seqs = np.append(0, seqs[:-1]).cumsum()
        bool_evid_idx = [ False if v<V else True for v in range(3*V) ]
        evid_idx = np.arange(V, 3*V)
        # The variables below are used to compute component marginals
        marg_idx_arr = np.array([False] * V * V).reshape(V, V)
        marg_idx_arr[ np.arange(V), np.arange(V) ] = True
        marg_vals_arr = np.ones((V, 1), dtype=int)
        # Create filenames
        base_dir = out_dir + '/' + dset_name
        base_filename = base_dir + '/' + dset_name + '.' + str(K) + '.' + str(L)
        pred_filenames, marg_filenames = [], []
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        def create_file(suffix, l, filenames_list):
            filename = base_filename + '.' + str(l) + '.' + suffix
            if os.path.exists(filename):
                os.remove(filename)
            filenames_list.append(filename)
        for l in range(L):
            create_file('predict', l+1, pred_filenames)
        create_file('marg', 0, marg_filenames)
        # Start particle filtering for each sequence
        for s in range(seqs.shape[0]):
            print('* Processing sequence ' + str(s+1) + ' of ' + \
                    str(N) + ' (size: ' + str(seqs[s]) + ')')
            print('p = 1')
            # Initialize explanations tensor for sequence
            all_explanations = np.zeros((seqs[s], L, V+1))
            # Initialize marginals tensor for sequence
            all_marginals = np.zeros((seqs[s], V))
            # Instantiate conditional cutset network
            curr_model = self.prior_model.convert_to_cnet(evid_dset[cum_seqs[s]:cum_seqs[s] + 1])
            # Compute marginals
            marginals = [ curr_model.prob_evid(curr_model.tree, marg_idx_arr[v], marg_vals_arr[v]) \
                        for v in range(V) ]
            all_marginals[0,:] = marginals
            # Generate a set of new, unique particles
            prev_particles = np.unique(curr_model.weighted_samples(K), axis=0)
            prev_weights = np.array([ -curr_model.prob_evid(curr_model.tree, np.array([ True ] * V),
                                                        particle) for particle in prev_particles ])
            prev_weights /= -prev_weights.sum() - 0.000000001
            # Compute explanations
            explanations = np.sort(np.concatenate((prev_weights[:,np.newaxis], prev_particles), axis=1), axis=0)[:L]
            all_explanations[0,:explanations.shape[0],:] = explanations
            # Filter remaining particles using transition model
            for i in range(cum_seqs[s] + 1, cum_seqs[s] + seqs[s]):
                p = i - cum_seqs[s]
                print('p = ' + str(p+1))
                # Initialize new particles and marginals
                new_particles = []
                new_marginals = np.zeros((len(prev_particles),V))
                # Generate new particles
                for k in range(len(prev_particles)):
                    # Get kth particle
                    particle = prev_particles[k]
                    # Create particle evidence vector
                    curr_evid = np.hstack((particle, evid_dset[i]))
                    # Instantiate particle model
                    curr_model = self.transition_model.convert_to_cnet(curr_evid[np.newaxis,:])
                    # Generate a set of new, unique particles
                    curr_particles = np.unique(curr_model.weighted_samples(K), axis=0)
                    curr_weights = np.array([ -curr_model.prob_evid(curr_model.tree,
                                                np.array([ True ] * V), particle) * -prev_weights[k]
                                                for particle in curr_particles ])
                    curr_margs = prev_weights[k] * np.array([ curr_model.prob_evid(curr_model.tree,
                                                                marg_idx_arr[v],
                                                                marg_vals_arr[v]) for v in range(V) ])
                    # Add to new particles and new marginals
                    new_particles.append([ np.concatenate((curr_weights[:,np.newaxis], curr_particles), axis=1) ])
                    new_marginals[k,:] = curr_margs
                # Normalize and compute marginals
                marginals = new_marginals.sum(axis=0) / prev_weights.sum() + 0.000000001
                all_marginals[p,:] = marginals
                # Assign new particles to previous particles
                prev_super_particles = np.unique(np.squeeze(np.concatenate(new_particles, axis=1), axis=0),axis=0)[:K,:]
                prev_weights, prev_particles = prev_super_particles[:,0], prev_super_particles[:,1:]
                prev_weights /= -prev_weights.sum() - 0.000000001
                # Compute explanations
                explanations = np.sort(prev_super_particles, axis=0)[:L]
                all_explanations[p,:explanations.shape[0],:] = explanations
            # Write explanations to files
            for l in range(L):
                with open(pred_filenames[l], "ab") as pred_file:
                    np.savetxt(pred_file, all_explanations[:,l,1:], fmt='%i', delimiter=',')
            with open(marg_filenames[0], "ab") as marg_file:
                np.savetxt(marg_file, all_marginals, fmt='%.2f', delimiter=',')
            return

if __name__ == '__main__':
    # Check if arguments are present
    if (len(sys.argv) < 7):
        print(usage_str)
        sys.exit(1)
    # Read in arguments
    data_dir = sys.argv[1]
    dset_name = sys.argv[2]
    max_depth_prior = int(sys.argv[3])
    max_depth = int(sys.argv[4])
    function = sys.argv[5].strip() if (sys.argv[5].strip() in ["LR", "NN"]) else "LR"
    alpha = float(sys.argv[6])
    K = int(sys.argv[7]) if (len(sys.argv)>7) else 10
    L = min(int(sys.argv[8]), K) if (len(sys.argv)>8) else 3
    out_dir = sys.argv[9] if (len(sys.argv)>9) else data_dir
    print("=============================")
    print("Explanation Layer Compilation")
    print("=============================")
    print("Data Directory: " + data_dir)
    print("Dataset Name: " + dset_name)
    print("Max Depth (Prior): " + str(max_depth_prior))
    print("Max Depth (Transition): " + str(max_depth))
    print("Function: " + function)
    print("Alpha: " + str(alpha))
    print("K: " + str(K))
    print("L: " + str(L))
    print("Output Directory: " + out_dir)
    # Load Data
    print("\n*** Loading data")
    infile_path = data_dir + '/' + dset_name + '/' + dset_name
    train_ground_dset = np.loadtxt(infile_path + ".train.ground", delimiter=',', dtype=int)
    train_evid_dset = np.loadtxt(infile_path + ".train.evid", delimiter=',', dtype=int)
    test_ground_dset = np.loadtxt(infile_path + ".test.ground", delimiter=',', dtype=int)
    test_evid_dset = np.loadtxt(infile_path + ".test.evid", delimiter=',', dtype=int)
    train_seqs = np.loadtxt(infile_path + ".train.seq", delimiter=',', ndmin=1, dtype=int)
    test_seqs = np.loadtxt(infile_path + ".test.seq", delimiter=',', ndmin=1, dtype=int)
    # Learning Model
    print("\n*** Learning model")
    dccn = ExpDCCN()
    dccn.learn(train_ground_dset, train_evid_dset, train_seqs, max_depth_prior,
                max_depth, function, alpha)
    # Compile knowledge using particle filtering
    print("\n*** Compiling knowledge using Particle Filtering (check " + \
                                            out_dir + '/' + dset_name + ")")
    dccn.pfCompile(test_evid_dset, test_seqs, out_dir, dset_name, K, L)
    # Show that program has finished
    print("=============================")
    print("***** Finished Program! *****")
    print("=============================")
