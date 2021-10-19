import numpy as np
import dcn.util
import sys

usage_str = """Usage:
======
python hmm.py <dir-path> <file-name>

Arguments:
==========
<dir-path>:\tPath to directory that contains the datasets
<file-name>:\tName of the dataset"""

class HMM:
    # Extract all domains
    @staticmethod
    def extract_all_doms(dset):
        if (dset.shape[0]==0):
            return []
        all_doms = []
        for c in range(dset.shape[1]):
            all_doms += [ np.unique(dset[:,c]) ]
        return all_doms

    # Prior probabilities
    @staticmethod
    def create_prior_probs(dset):
        if (dset.shape[0]==0):
            return []
        max_dom_size = 0
        for c in range(dset.shape[1]):
            max_dom_size = max(max_dom_size, np.unique(dset[:,c]).size)
        smooth_factor = (dset.shape[0]*0.001) / max_dom_size
        all_probs = []
        for c in range(dset.shape[1]):
            probs = np.unique(dset[:,c], return_counts=True)[1].astype(float) + \
                    smooth_factor
            all_probs += [ probs / probs.sum() ]
        return all_probs

    # Transition Dataset
    @staticmethod
    def create_transition_dset(dset, seqs):
        if (dset.shape[0]==0 or seqs.sum()!=dset.shape[0]):
            return np.array([])
        all_indices = np.arange(dset.shape[0])
        curr_indices = np.setdiff1d(all_indices,np.cumsum(seqs)[:-1])[1:]
        prev_indices = np.setdiff1d(all_indices,np.cumsum(seqs)-1)
        return np.hstack((dset[prev_indices,:],dset[curr_indices,:]))

    # Conditional probabilities P(curr|prev)
    @staticmethod
    def create_transition_cpt(prev_column, curr_column, doms):
        # doms = np.unique(prev_column)
        cond_probs = np.zeros((doms.size,doms.size))
        if (doms.size<np.unique(curr_column).size):
            return np.array([]).astype(float)
        smooth_factor = (prev_column.shape[0]*0.001) / doms.size
        for i in range(doms.size):
            for j in range(doms.size):
                cond_probs[i,j] = ((curr_column==doms[i])*(prev_column==doms[j])).sum() + \
                                    smooth_factor
        z = cond_probs.sum(axis=0).reshape(1,doms.size)
        return (cond_probs / z).flatten()

    # Find all conditional probabilities for a given dataset
    @staticmethod
    def create_transition_cpts(dset, seqs, all_doms):
        if (dset.shape[0]==0 or len(all_doms) != dset.shape[1]):
            return []
        transition_dset = create_transition_dset(dset, seqs)
        all_cpts = []
        for c in range(dset.shape[1]):
            prev_column = transition_dset[:,c]
            curr_column = transition_dset[:,c+dset.shape[1]]
            all_cpts += [ create_transition_cpt(prev_column, curr_column, all_doms[c]) ]
        return all_cpts

    # Multiply msg with CPT
    @staticmethod
    def multiply_msg_with_cpt(msg, transition):
        if (msg.shape[0]**2 != transition.shape[0]):
            return np.array([], dtype=float)
        n = msg.shape[0]
        return (transition.reshape(n,n) * msg.reshape(1,n)).sum(axis=1)

    # Multiply msgs with CPTs
    @staticmethod
    def multiply_msgs_with_cpt(msgs, all_cpts):
        if (len(msgs) != len(all_cpts)):
            return []
        new_msgs = []
        for i in range(len(msgs)):
            new_msgs += [ multiply_msg_with_cpt(msgs[i], all_cpts[i]) ]
        return new_msgs

    # Compute LL for a single data point with respect to a set of messages
    @staticmethod
    def LL(row, all_doms, all_msgs):
        ll = 0.0
        for v in range(row.shape[0]):
            d = np.where(all_doms[v]==row[v])[0][0]
            ll += np.log(all_msgs[v][d])
        return ll

    # Compute average log likelihood
    @staticmethod
    def avg_LL(dset, seqs, all_doms, all_probs, all_cpts, evid_idx=np.array([]), debug=True):
        if(dset.shape[0]==0 or len(all_doms)!=len(all_probs) or \
            len(all_probs)!=len(all_cpts) or dset.shape[0]!=seqs.sum()):
            return 1.0
        total_ll = 0.0
        V = len(all_doms)
        cum_seqs = np.append(0, seqs[:-1]).cumsum()
        evid_idx = np.arange(dset.shape[1]) if not evid_idx.size else evid_idx
        for s in range(seqs.shape[0]):
            if (debug): print("Processing sequence " + str(s + 1) + " out of " + str(seqs.size))
            curr_data = np.zeros(V * 2, dtype=int)
            for i in range(cum_seqs[s], cum_seqs[s]+seqs[s]):
                cpts = all_cpts if i > cum_seqs[s] else all_probs
                curr_data[V:] = dset[i]
                local_ll = 0.0
            # for v in range(V):
            for v in evid_idx:
                addr1 = np.where(all_doms[v] == curr_data[v])[0][0]
                addr2 = (addr1*all_doms[v].shape[0]) + np.where(all_doms[v] == curr_data[V+v])[0][0]
                local_ll += np.log(cpts[v][addr2]) if i>0 else np.log(cpts[v][addr2])
            if (debug): print("i=" + str(i + 1) + ": " + str(local_ll))
            curr_data[:V] = dset[i]
            total_ll += local_ll
        return total_ll / dset.shape[0]

    @staticmethod
    def avg_CLL(dset, seqs, evid_idx, all_doms, all_probs, all_cpts, debug=True):
        if(dset.shape[0]==0 or len(all_doms)!=len(all_probs) or \
            len(all_probs)!=len(all_cpts) or dset.shape[0]!=seqs.sum() or \
            ((evid_idx<0)*(evid_idx>dset.shape[1])).sum()):
            return 1.0
        total_cll = 0.0
        V = len(all_doms)
        cum_seqs = np.append(0, seqs[:-1]).cumsum()
        all_idx = np.arange(dset.shape[1])
        non_evid_idx = np.setdiff1d(all_idx, evid_idx)
        for s in range(seqs.shape[0]):
            if (debug): print("Processing sequence " + str(s + 1) + " out of " + str(seqs.size))
            curr_data = np.zeros(V*2, dtype=int)
            msg_probs = all_probs
            for i in range(cum_seqs[s], cum_seqs[s]+seqs[s]):
                local_cll = 0.0
                for v in non_evid_idx:
                    addr = np.where(all_doms[v] == curr_data[v])[0][0]
                    # Compute local_cll
                    local_cll += np.log(msg_probs[v][addr])
                    # Compute next message
                    D = all_doms[v].size
                    msg_probs[v] = (all_cpts[v].reshape(D,D) * msg_probs[v].reshape(D,1)).sum(axis=0)
                if (debug): print("i=" + str(i + 1) + ": " + str(local_cll))
                total_cll += local_cll
                curr_data[:V] = dset[i]
        return total_cll / dset.shape[0]

    @staticmethod
    def hmm_inference(dir_name, file_name, evid_idx, dom_vals=[]):
        # Initialize variables
        infile_path = dir_name + '/' + file_name
        # Load datasets
        train_dset = np.loadtxt(infile_path + ".train", delimiter=',', dtype=int)
        test_dset = np.loadtxt(infile_path + ".test", delimiter=',', dtype=int)
        cmll_dset = np.loadtxt(infile_path + ".cmll", delimiter=',', dtype=int)
        train_seqs = np.loadtxt(infile_path + ".train.seq", delimiter=',', ndmin=1, dtype=int)
        test_seqs = np.loadtxt(infile_path + ".test.seq", delimiter=',', ndmin=1, dtype=int)
        cmll_seqs = np.loadtxt(infile_path + ".cmll.seq", delimiter=',', ndmin=1, dtype=int)
        final_test_dset = test_dset if not evid_idx.size else cmll_dset
        final_test_seqs = test_seqs if not evid_idx.size else cmll_seqs
        # Train model
        V = train_dset.shape[1]
        all_scopes = [ np.array([x]) for x in range(V) ] + \
                        [ np.array([x,x+V]) for x in range(V) ]
        all_doms = extract_all_doms(train_dset) if (len(dom_vals) != V) else dom_vals
        all_probs = create_prior_probs(train_dset)
        # all_cpts = create_transition_cpts(train_dset, train_seqs, all_doms)
        transition_dset = create_transition_dset(train_dset, train_seqs)
        all_cpts = util.learn_model_params(transition_dset, all_doms*2, all_scopes)[V:]
        # Return average log-likelihood
        return avg_LL(final_test_dset, final_test_seqs, all_doms, all_probs, all_cpts, evid_idx)

if __name__ == "__main__":
    # Check if arguments are present
    if(len(sys.argv)<3):
        print(usage_str)
        sys.exit(1)
    # Read in arguments
    dir_name = sys.argv[1]
    file_name = sys.argv[2]
    dom_vals = [vals for vals in np.loadtxt(dir_name + "/" + file_name + ".doms", skiprows=1,
                                            delimiter=",", dtype=int)]
    evid_idx = np.array([]) if len(sys.argv)<4 else np.fromstring(sys.argv[3], dtype=int, sep=",")
    # Perform inference
    print("==========================")
    print("HMM Inference")
    print("==========================")
    print("Directory: " + dir_name)
    print("File Name: " + file_name)
    print("Evidence indices: " + str(evid_idx))
    ll = HMM.hmm_inference(dir_name,file_name,evid_idx)
    print("Avg LL: " + str(ll))

# python hmm.py "/Users/chiradeep/dev/cdnets/data/synth" synth.p.2.2
