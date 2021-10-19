import numpy as np
import cPickle as pickle
import hmm
import cdnets
import util
import os
import sys

usage_str = """Usage:
======
python DRCN.py <dir-path> <model-path> <file-name> <type> <prior-depth> <trans-cond-depth> <trans-cn-depth> [ <b> <K> <M> <maxiter> <evid-idx> ]

Arguments:
==========
<dir-path>:\tPath to directory that contains the datasets
<model-path>:\tPath to directory where model will be stored
<file-name>:\tName of the dataset
<type>:\tType of model ("rf": Random Forest, "arf":, AND Random Forest)
<prior-depth>:\tMaximum depth of prior cutset network
<trans-cond-depth>:\tMaximum depth of decision tree in transition CDNet
<trans-cn-depth>:\tMaximum depth of leaf cutset network in transition CDNet
<b>:\tb parameter for adaptive partitions (default: 0.5)
<K>:\tNumber of trees in forest (default: 10)
<M>:\tNumber of bootstrap samples per tree (default: size of dataset)
<maxiter>:\tMaximum number of iterations for EM (default: 10)
<evid-idx>:\tEvidence indices for computing CLL (optional)"""

class RandomForestCNet:
    def __init__(self):
        self.K = 0
        self.size = 0
        self.probs = np.array([])
        self.trees = []
        self.nleaves = 0
    def learn(self, dset, max_depth, K=10, M=-1, maxiter=10, dom_vals=[], debug=True):
        # Set parameters
        self.K = K
        self.size = dset.shape[1]
        self.probs = np.ones(K)
        M = dset.shape[0] if M<=0 else M
        N = dset.shape[0]
        # Learn K CNets
        for k in range(K):
            if (debug): print("Learning tree " + str(k+1) + " of " + str(K))
            bootstrap_dset = dset[np.random.choice(N,M),:]
            cnet = cdnets.CNet()
            cnet.learn(bootstrap_dset, max_depth, 1, dom_vals, random_flag=True)
            self.trees += [ cnet ]
            self.probs[k] = np.exp(cnet.avgLL(dset))
            self.nleaves += 1
        # Normalize probs
        self.probs /= 1.0 if self.probs.sum()==0 else self.probs.sum()
        # Now use, EM to improve the model
        prev_train_ll = 0.0
        for i in range(maxiter):
            if (debug): print("EM iteration " + str(i+1) + " of " + str(maxiter))
            # E-Step
            joint_weights = np.exp(self.compLL(dset)) * self.probs
            joint_weights = joint_weights[joint_weights.sum(axis=1)>0.0]
            post_weights = joint_weights / joint_weights.sum(axis=1)[:,np.newaxis]
            self.probs = post_weights.sum(axis=0) / dset.shape[0]
            # Print likelihood
            train_ll = np.log(joint_weights.sum(axis=1)).mean()
            if (debug): print("Training Likelihood: " + str(train_ll))
            # M-Step
            for k in range(self.K):
                self.trees[k].learn(dset, max_depth, 1, dom_vals, post_weights[:,k])
            # Check for convergence
            if abs(train_ll - prev_train_ll) <= 0.001:
                break
            prev_train_ll = train_ll
    def probEvid(self, evid_idx, evid_vals):
        # Return 1 if no evidence
        if (evid_idx.size==0):
            return 1.0
        # Validation
        if (evid_idx.ndim!=1 or evid_vals.ndim!=1 or evid_idx.size!=evid_vals.size):
            return -1.0
        # Compute probability of evidence
        cond_probs = np.array([ self.trees[k].probEvid(evid_idx, evid_vals) for k in range(self.K) ])
        if ((cond_probs<0.0)*(cond_probs>1.0)).sum()==1.0:
            return -1.0
        probs = self.probs * cond_probs
        return probs.sum()
    def compLL(self, dset):
        # Validation
        if (dset.ndim != 2 or dset.shape[0] == 0):
            return np.array([[]])
        # Compute LLs
        ll_arr = np.zeros((dset.shape[0], self.K))
        for i in range(dset.shape[0]):
            ll_arr[i,:] = np.array([ self.trees[k].avgLL(dset[i:i+1,:]) for k in range(self.K) ])
        # Return LL
        return ll_arr
    def AvgCompLL(self, dset):
        return self.compLL(dset).sum(axis=0) / dset.shape[0]
    def LL(self, dset):
        comp_probs = np.exp(self.compLL(dset))
        return np.log((self.probs * comp_probs).sum(axis=1))
    def avgLL(self, dset):
        return self.LL(dset).sum() / dset.shape[0]

class RandomForestCDNet:
    def __init__(self):
        self.K = 0
        self.size = 0
        self.probs = np.array([])
        self.trees = []
        self.x_indices = np.array([])
        self.nleaves = 0
    def learn(self, dset, x_indices, max_cond_depth, max_depth, K=10, M=-1,
              maxiter=10, dom_vals=[], debug=True):
        # Set parameters
        self.K = K
        self.size = dset.shape[1]
        self.probs = np.ones(K)
        self.x_indices = x_indices
        M = dset.shape[0] if M<=0 else M
        N = dset.shape[0]
        # Learn K CDNets
        for k in range(K):
            if (debug): print("Learning tree " + str(k+1) + " of " + str(K))
            bootstrap_dset = dset[np.random.choice(N,M),:]
            cdnet = cdnets.CDNet()
            cdnet.learn(bootstrap_dset, x_indices, max_cond_depth, max_depth, 1, dom_vals, random_flag=True)
            self.trees += [ cdnet ]
            self.probs[k] = np.exp(cdnet.avgLL(dset))
            self.nleaves += cdnet.nleaves
        # Normalize probs
        self.probs /= 1.0 if self.probs.sum()==0 else self.probs.sum()
        # Now use, EM to improve the model
        prev_train_ll = 0.0
        for i in range(maxiter):
            if (debug): print("EM iteration " + str(i+1) + " of " + str(maxiter))
            # E-Step
            joint_weights = np.exp(self.compLL(dset)) * self.probs
            joint_weights = joint_weights[joint_weights.sum(axis=1)>0.0]
            post_weights = joint_weights / joint_weights.sum(axis=1)[:,np.newaxis]
            self.probs = post_weights.sum(axis=0) / dset.shape[0]
            # Print likelihood
            train_ll = np.log(joint_weights.sum(axis=1)).mean()
            if (debug): print("Training Likelihood: " + str(train_ll))
            # M-Step
            for k in range(self.K):
                self.trees[k].updateParams(dset, max_depth, 1, dom_vals, post_weights[:,k])
            # Check for convergence
            if abs(train_ll - prev_train_ll) <= 0.001:
                break
            prev_train_ll = train_ll
    def multiplyAndMarginalize(self, x_dist, evid_idx=np.array([]),
                                evid_vals=np.array([]), idx_replace=np.array([])):
        # Basic Validation
        if (evid_idx.ndim != 1 or evid_vals.ndim!=1 or evid_idx.size != evid_vals.size or \
                x_dist.size != self.x_indices.size or evid_idx.size > x_dist.size):
            return RandomForestCNet()
        # Create message
        msg = RandomForestCNet()
        msg.K = 0
        msg.size = x_dist.size
        msg.nleaves = 0
        # Perform multiplication and marginalization
        for k in range(self.K):
            mixcnet = self.trees[k].multiplyAndMarginalize(x_dist, evid_idx, evid_vals, idx_replace)
            msg.K += mixcnet.probs.size
            msg.nleaves += mixcnet.probs.size
            msg.probs = np.concatenate((msg.probs, self.probs[k]*mixcnet.probs))
            msg.trees += mixcnet.cnets
        # Return message
        return msg
    def probEvid(self, evid_idx, evid_vals):
        # Validation
        if (evid_idx.ndim!=1 or evid_vals.ndim!=1 or evid_idx.size!=evid_vals.size or \
                evid_idx.size==0):
            return -1.0
        # Compute probability of evidence
        cond_probs = np.array([ self.trees[k].probEvid(evid_idx, evid_vals) for k in range(self.K) ])
        if ((cond_probs<0.0)*(cond_probs>1.0)).sum()==1.0:
            return -1.0
        probs = self.probs * cond_probs
        return probs.sum()
    def compLL(self, dset):
        # Validation
        if (dset.ndim != 2 or dset.shape[0] == 0):
            return np.array([[]])
        # Compute LLs
        ll_arr = np.zeros((dset.shape[0], self.K))
        for i in range(dset.shape[0]):
            ll_arr[i,:] = np.array([ self.trees[k].avgLL(dset[i:i+1,:]) for k in range(self.K) ])
        # Return LL
        return ll_arr
    def AvgCompLL(self, dset):
        return self.compLL(dset).sum(axis=0) / dset.shape[0]
    def LL(self, dset):
        comp_probs = np.exp(self.compLL(dset))
        total_ll = np.log((self.probs * comp_probs).sum(axis=1))
        total_ll[(total_ll<1.0)*(total_ll>0.0)] = 0.0
        return total_ll
    def avgLL(self, dset):
        dummy = self.LL(dset).sum()
        return dummy / dset.shape[0]

# This model is more expressive, but intractable
class AndRFCDNet:
    def __init__(self):
        self.rfcdnets = []
        self.partitions = []
        self.x_indices = np.array([])
        self.size = 0
        self.nleaves = 0
    def learn(self, dset, x_indices, max_cond_depth, max_depth, K=10, M=-1,
              maxiter=10, dom_vals=[], debug=True):
        # Validation
        if (dset.ndim != 2 or max_cond_depth < 0 or (len(dom_vals) % dset.shape[1]) != 0 or \
                (x_indices.size >= dset.shape[1]) or (x_indices >= dset.shape[1]).sum()):
            return
        # Initialization
        all_idx = np.arange(dset.shape[1])
        y_idx = np.setdiff1d(all_idx, x_indices)
        self.size = dset.shape[1]
        self.x_indices = x_indices
        self.partitions = [ np.array([i]) for i in y_idx ]
        dom_vals = dom_vals if (dset.shape[1] == len(dom_vals)) else hmm.extract_all_doms(dset)
        # Learn RFCDNets
        for p in range(len(self.partitions)):
            if (debug): print("* Learning partition "+str(p+1)+" of "+str(len(self.partitions)))
            partition = self.partitions[p]
            to_keep_idx = np.union1d(x_indices, partition)
            new_x_indices = np.nonzero(np.in1d(to_keep_idx, x_indices))[0]
            new_dset = dset[:, to_keep_idx]
            new_dom_vals = [dom_val for dom_val in np.array(dom_vals)[to_keep_idx]]
            rfcdnet = RandomForestCDNet()
            self.rfcdnets += [rfcdnet]
            # Learn RFCDNet structure and parameters
            rfcdnet.learn(new_dset, new_x_indices, max_cond_depth,
                        max_depth, K, M, maxiter, new_dom_vals)
            self.nleaves += rfcdnet.nleaves
    def LL(self, row):
        if (row.ndim != 1 or row.size != self.size):
            return 1.0
        total_ll = 0.0
        for p in range(len(self.partitions)):
            partition_scope = np.union1d(self.x_indices, self.partitions[p])
            new_row = row[partition_scope]
            partition_ll = self.rfcdnets[p].avgLL(new_row.reshape(1,new_row.size))
            if (partition_ll>0.0):
                return 1.0
            total_ll += partition_ll
        return total_ll
    def avgLL(self, dset):
        if (dset.ndim != 2 or dset.shape[1] != self.size or dset.shape[0] == 0):
            return 1.0
        total_ll = 0.0
        for i in range(dset.shape[0]):
            local_ll = self.LL(dset[i, :])
            if (local_ll > 0.0):
                return 1.0
            total_ll += local_ll
        return total_ll / dset.shape[0]

# This model is less expressive, but tractable
class TAndRFCDNet:
    def __init__(self):
        self.rfcdnets = []
        self.x_partitions = []
        self.y_partitions = []
        self.local_x_indices = []
        self.size = 0
        self.nleaves = 0
    def learn(self, dset, x_indices, max_cond_depth, max_depth, b=0.5, K=10,
              M=-1, maxiter=10, dom_vals=[], partitions=[], debug=True):
        # General Validation
        if (dset.ndim != 2 or max_cond_depth < 0 or (len(dom_vals) % dset.shape[1]) != 0 or \
                (2*x_indices.size) != dset.shape[1] or (x_indices >= dset.shape[1]).sum()):
            return
        # Validate partitions
        all_part_vars = np.concatenate(partitions) if len(partitions)>0 else np.array([])
        y_size = dset.shape[1]-x_indices.size
        if len(partitions)>0 and (all_part_vars.size!=y_size or \
            np.setdiff1d(np.arange(y_size),all_part_vars).size>0):
            return
        # Initialization
        all_idx = np.arange(dset.shape[1])
        y_idx = np.setdiff1d(all_idx, x_indices)
        self.size = dset.shape[1]
        dom_vals = dom_vals if (dset.shape[1] == len(dom_vals)) else hmm.extract_all_doms(dset)
        # Generate partitions
        y_mi_matrix = util.gen_mi_matrix(dset[:,y_idx])
        y_partitions = util.gen_adaptive_partitions(y_mi_matrix, dset.shape[0], b) if len(partitions)==0 else partitions
        self.x_partitions = [ x_indices[y_part] for y_part in y_partitions ]
        self.y_partitions = [ y_idx[y_part] for y_part in y_partitions ]
        # Learn RFCDNets
        for p in range(len(self.x_partitions)):
            if (debug): print("* Learning partition "+str(p+1)+" of "+str(len(self.x_partitions)))
            part_idx = np.union1d(self.x_partitions[p], self.y_partitions[p])
            local_x_idx = np.nonzero(np.in1d(part_idx, self.x_partitions[p]))[0]
            self.local_x_indices += [ local_x_idx ]
            local_dset = dset[:, part_idx]
            local_dom_vals = [ dom_val for dom_val in np.array(dom_vals)[part_idx] ]
            rfcdnet = RandomForestCDNet()
            self.rfcdnets += [rfcdnet]
            # Learn RFCDNet structure and parameters
            rfcdnet.learn(local_dset, local_x_idx, max_cond_depth,
                        max_depth, K, M, maxiter, local_dom_vals)
            self.nleaves += rfcdnet.nleaves
    def multiplyAndMarginalize(self, x_dist, evid_idx=np.array([]),
                                evid_vals=np.array([]), idx_replace=np.array([])):
        # Basic Validation
        if (evid_idx.ndim != 1 or evid_vals.ndim!=1 or evid_idx.size != evid_vals.size or \
                x_dist.size != np.concatenate(self.x_partitions).size or \
                evid_idx.size > x_dist.size or len(x_dist.partitions) != len(self.y_partitions)):
            return TAndRFCNet()
        # Create message template
        msg = TAndRFCNet()
        msg.partitions = x_dist.partitions
        msg.size = x_dist.size
        # Multiply and marginalize
        for p in range(len(x_dist.partitions)):
            local_idx_map = np.nonzero(np.in1d(evid_idx,x_dist.partitions[p]))[0]
            new_evid_idx = np.arange(local_idx_map.size)
            new_evid_vals = evid_vals[local_idx_map]
            new_idx_replace = idx_replace[local_idx_map] if idx_replace.size else np.array([])
            # new_idx_replace = np.concatenate((self.x_partitions[p], self.y_partitions[p]))
            rfcnet = self.rfcdnets[p].multiplyAndMarginalize(x_dist.rfcnets[p], new_evid_idx,
                                                            new_evid_vals, new_idx_replace)
            msg.rfcnets += [ rfcnet ]
            msg.nleaves += rfcnet.nleaves
        # Return new message
        return msg
    def LL(self, row):
        if (row.ndim != 1 or row.size != self.size):
            return 1.0
        total_ll = 0.0
        for p in range(len(self.x_partitions)):
            partition_scope = np.union1d(self.x_partitions[p], self.y_partitions[p])
            new_row = row[partition_scope]
            partition_ll = self.rfcdnets[p].avgLL(new_row.reshape(1,new_row.size))
            if (partition_ll>0.0):
                return 1.0
            total_ll += partition_ll
        return total_ll
    def avgLL(self, dset):
        if (dset.ndim != 2 or dset.shape[1] != self.size or dset.shape[0] == 0):
            return 1.0
        total_ll = 0.0
        for i in range(dset.shape[0]):
            local_ll = self.LL(dset[i, :])
            if (local_ll > 0.0):
                return 1.0
            total_ll += local_ll
        return total_ll / dset.shape[0]

# CNET version of TAndRFCDNet
class TAndRFCNet:
    def __init__(self):
        self.rfcnets = []
        self.partitions = []
        self.size = 0
        self.nleaves = 0
    def learn(self, dset, max_depth, b=0.5, K=10, M=-1, maxiter=10, dom_vals=[], partitions=[], debug=True):
        # Validation
        if (dset.ndim != 2 or max_depth < 0 or (len(dom_vals) % dset.shape[1]) != 0):
            return
        # Validate partitions
        all_part_vars = np.concatenate(partitions) if len(partitions) > 0 else np.array([])
        y_size = dset.shape[1]
        if len(partitions) > 0 and (all_part_vars.size != y_size or \
            np.setdiff1d(np.arange(y_size), all_part_vars).size > 0):
            return
        # Initialization
        all_idx = np.arange(dset.shape[1])
        self.size = dset.shape[1]
        dom_vals = dom_vals if (dset.shape[1] == len(dom_vals)) else hmm.extract_all_doms(dset)
        # Generate partitions
        mi_matrix = util.gen_mi_matrix(dset)
        self.partitions = util.gen_adaptive_partitions(mi_matrix, dset.shape[0], b) if len(partitions)==0 else partitions
        # Learn RFCDNets
        for p in range(len(self.partitions)):
            if (debug): print("* Learning partition "+str(p+1)+" of "+str(len(self.partitions)))
            part_idx = self.partitions[p]
            local_dset = dset[:, part_idx]
            local_dom_vals = [ dom_val for dom_val in np.array(dom_vals)[part_idx] ]
            rfcnet = RandomForestCNet()
            self.rfcnets += [ rfcnet ]
            # Learn RFCDNet structure and parameters
            rfcnet.learn(local_dset, max_depth, K, M, maxiter, local_dom_vals)
            self.nleaves += rfcnet.nleaves
    def probEvid(self, evid_idx, evid_vals):
        # Validation
        if (evid_idx.ndim != 1 or evid_vals.ndim != 1 or evid_idx.size != evid_vals.size or
                evid_idx.size == 0):
            return -1.0
        # Compute probability of evidence
        prob = 1.0
        for p in range(len(self.partitions)):
            partition_scope = self.partitions[p]
            scope_to_evid_map = np.nonzero(np.in1d(partition_scope, evid_idx))
            evid_to_scope_map = np.nonzero(np.in1d(evid_idx, partition_scope))
            local_evid_idx = np.arange(partition_scope.size)[scope_to_evid_map]
            local_evid_vals = evid_vals[evid_to_scope_map]
            local_prob = self.rfcnets[p].probEvid(local_evid_idx, local_evid_vals)
            if (local_prob<0.0):
                return -1.0
            prob *= local_prob
        return prob
    def LL(self, row):
        if (row.ndim != 1 or row.size != self.size):
            return 1.0
        total_ll = 0.0
        for p in range(len(self.partitions)):
            partition_scope = self.partitions[p]
            new_row = row[partition_scope]
            partition_ll = self.rfcnets[p].avgLL(new_row.reshape(1,new_row.size))
            if (partition_ll>0.0):
                return 1.0
            total_ll += partition_ll
        return total_ll
    def avgLL(self, dset):
        if (dset.ndim != 2 or dset.shape[1] != self.size or dset.shape[0] == 0):
            return 1.0
        total_ll = 0.0
        for i in range(dset.shape[0]):
            local_ll = self.LL(dset[i, :])
            if (local_ll > 0.0):
                return 1.0
            total_ll += local_ll
        return total_ll / dset.shape[0]

class DRCN:
    def __init__(self):
        self.prior_model = None
        self.transition_model = None
    def learn(self, dset, seqs, type, dset_name, max_prior_depth=0,
                max_trans_cond_depth=1, max_trans_cn_depth=0, b=0.5, K=10,
                M=-1, maxiter=10, model_dir="", dom_vals=[], debug=True):
        # Validation
        if dset.ndim !=2 or seqs.ndim!=1 or dset.shape[0] != seqs.sum() or max_prior_depth<0 or \
            max_trans_cond_depth<0 or max_trans_cn_depth<0 or K<0 or M<-1:
            return
        # Load model, if it exists
        file_name = type + "." + dset_name + "." + str(max_prior_depth) + "." + \
                    str(max_trans_cond_depth) + "." + str(max_trans_cn_depth) + "." + \
                    str(b) + "." + str(K) + "." + str(maxiter)
        if os.path.exists(model_dir + "/" + file_name + ".model"):
            self.load(model_dir, file_name)
            return
        # Generate partitions
        L = 2
        C = np.ceil((1.0*dset.shape[1])/L).astype(int)
        # C = 20
        mi_matrix = util.gen_mi_matrix(dset)
        # partitions = util.gen_clt_partitions(mi_matrix+0.00000001, L, np.arange(dset.shape[1]))
        # partitions = util.gen_clustered_partitions(mi_matrix, C, L)
        partitions = util.gen_adaptive_partitions(mi_matrix, dset.shape[0], b)
        # partitions = np.array_split(np.arange(dset.shape[1]),13)
        # partitions = np.array_split(np.arange(6),2) + [ np.array([6]), np.array([7,8,9,10]) ]
        # Learn Prior Model
        print("** Learning prior model")
        cum_seqs = np.append(0, seqs[:-1]).cumsum()
        if type == "rf":
            self.prior_model = RandomForestCNet()
            self.prior_model.learn(dset, max_prior_depth, K, M,
                                   maxiter, dom_vals)
        else:
            self.prior_model = TAndRFCNet()
            self.prior_model.learn(dset, max_prior_depth, b, K, M,
                                  maxiter, dom_vals, partitions)
        # self.prior_model.learn(dset, max_prior_depth, b, K, M, maxiter, dom_vals, partitions)
        # Learn Transition Model
        print("** Learning transition model")
        transition_dset = hmm.create_transition_dset(train_dset, train_seqs)
        x_indices = np.arange(train_dset.shape[1])
        if type=="rf":
            self.transition_model = RandomForestCDNet()
            self.transition_model.learn(transition_dset, x_indices, max_trans_cond_depth,
                                        max_trans_cn_depth, K, M, maxiter, 2*dom_vals, debug)
        else:
            self.transition_model = TAndRFCDNet()
            self.transition_model.learn(transition_dset, x_indices, max_trans_cond_depth,
                                        max_trans_cn_depth, b, K, M, maxiter, 2*dom_vals, partitions, debug)
        # Save model if model directory exists
        if os.path.exists(model_dir):
            self.save(model_dir, file_name)
    def avgLL(self, dset, seqs, debug=True):
        # Validation
        if (dset.ndim != 2 or dset.shape[0] == 0 or seqs.ndim != 1 or dset.shape[0] != seqs.sum()):
            return 1.0
        # Initialize Variables
        total_ll = 0.0
        cum_seqs = np.append(0, seqs[:-1]).cumsum()
        for s in range(seqs.shape[0]):
            if (debug): print("Processing sequence " + str(s + 1) + " out of " + str(seqs.size))
            local_ll = self.prior_model.avgLL(dset[cum_seqs[s]:cum_seqs[s]+1])
            if (local_ll > 0.0):
                return 1.0
            total_ll += local_ll
            if (debug): print("i=" + str(cum_seqs[s] + 1) + ": " + str(local_ll))
            for i in range(cum_seqs[s] + 1, cum_seqs[s] + seqs[s]):
                local_ll = self.transition_model.avgLL(np.hstack((dset[i - 1:i], dset[i:i+1])))
                if (local_ll > 0.0):
                    return 1.0
                if (debug): print("i=" + str(i + 1) + ": " + str(local_ll))
                total_ll += local_ll
        # Return average LL
        return total_ll / dset.shape[0]
    def avgELL(self, dset, seqs, evid_idx, debug=True):
        # Validation
        if (dset.ndim != 2 or dset.shape[0] == 0 or seqs.ndim != 1 or dset.shape[0] != seqs.sum()):
            return 1.0
        # Initialize Variables
        total_ell = 0.0
        cum_seqs = np.append(0, seqs[:-1]).cumsum()
        for s in range(seqs.shape[0]):
            if (debug): print("Processing sequence " + str(s + 1) + " out of " + str(seqs.size))
            local_ell = np.log(self.prior_model.probEvid(evid_idx, dset[cum_seqs[s], evid_idx]))
            if (local_ell > 0.0):
                return 1.0
            if (debug): print("i=" + str(cum_seqs[s] + 1) + ": " + str(local_ell))
            total_ell += local_ell
            msg = self.transition_model.multiplyAndMarginalize(self.prior_model, evid_idx, dset[cum_seqs[s], evid_idx])
            # [DEBUG] Check if distribution is valid
            curr_assign = np.zeros(dset.shape[1], dtype=int)
            curr_assign[evid_idx] = dset[cum_seqs[s]+1, evid_idx]
            doms = np.full(dset.shape[1], 2)
            non_evid_idx = np.setdiff1d(np.arange(dset.shape[1]), evid_idx)
            new_prob = msg.probEvid(evid_idx, dset[cum_seqs[s]+1, evid_idx])
            old_prob = 0.0
            full_assign = np.zeros(dset.shape[1]*2, dtype=int)
            full_assign[evid_idx] = dset[cum_seqs[s], evid_idx]
            full_assign[dset.shape[1]:] = curr_assign
            for i in range(doms[non_evid_idx].prod()):
                for j in range(doms[non_evid_idx].prod()):
                    old_trans_prob = np.exp(self.transition_model.LL(full_assign))
                    old_prior_prob = self.prior_model.probEvid(non_evid_idx, full_assign[non_evid_idx])
                    print(str(full_assign) + " : " + str(old_prior_prob) + "/" + str(old_trans_prob) + "/" + str(old_prior_prob*old_trans_prob))
                    old_prob += old_prior_prob * old_trans_prob
                    full_assign[non_evid_idx] = util.calc_next_assign(full_assign[non_evid_idx], doms[non_evid_idx])
                full_assign[dset.shape[1] + non_evid_idx] = util.calc_next_assign(full_assign[dset.shape[1] + non_evid_idx], doms[non_evid_idx])
            print(str(old_prob) + " / " + str(new_prob))
            # [DEBUG] End
            for i in range(cum_seqs[s] + 1, cum_seqs[s] + seqs[s]):
                local_ell = np.log(msg.probEvid(evid_idx, dset[i, evid_idx]))
                if (local_ell > 0.0):
                    return 1.0
                if (debug): print("i=" + str(i + 1) + ": " + str(local_ell))
                prev_msg = msg
                total_ell += local_ell
                msg = self.transition_model.multiplyAndMarginalize(prev_msg, evid_idx, dset[i, evid_idx])
        # Return average CLL
        return total_ell / dset.shape[0]
    def save(self, dir_name, file_name):
        outfilename = dir_name + '/' + file_name + ".model"
        with open(outfilename,'wb') as outfile:
            pickle.dump(self,outfile,pickle.HIGHEST_PROTOCOL)
    def load(self, dir_name, file_name):
        infilename = dir_name + '/' + file_name + ".model"
        with open(infilename,'rb') as infile:
            obj = pickle.load(infile)
            self.prior_model = obj.prior_model
            self.transition_model = obj.transition_model

if __name__=="__main__":
    # Check if arguments are present
    if (len(sys.argv) < 8):
        print(usage_str)
        sys.exit(1)
    # Read in arguments
    dir_name = sys.argv[1]
    model_path = sys.argv[2]
    file_name = sys.argv[3]
    type = sys.argv[4]
    prior_depth = int(sys.argv[5])
    trans_cond_depth = int(sys.argv[6])
    trans_cn_depth = int(sys.argv[7])
    b = float(sys.argv[8]) if (len(sys.argv) > 8) else 0.5
    K = int(sys.argv[9]) if (len(sys.argv) > 9) else 10
    M = int(sys.argv[10]) if (len(sys.argv) > 10) else -1
    maxiter = int(sys.argv[11]) if (len(sys.argv) > 11) else 10
    evid_idx = np.fromstring(sys.argv[12], sep=",", dtype=int) if (len(sys.argv) > 12) else np.array([])
    print("==========================")
    print("DRCN Inference")
    print("==========================")
    print("Input Directory: " + dir_name)
    print("Model Directory: " + model_path)
    print("File Name: " + file_name)
    print("Model type: " + type)
    print("Max Depth (Prior): " + str(prior_depth))
    print("Max Conditional Depth (Transition): " + str(trans_cond_depth))
    print("Max CN Depth (Transition): " + str(trans_cn_depth))
    print("Decay parameter for adaptive thresholding (b): " + str(b))
    print("Number of trees (K): " + str(K))
    print("Number of boostrap samples (M): " + str(M))
    print("Max Iterations (for EM) : " + str(maxiter))
    print("Evidence Indices: " + str(evid_idx))
    # Load datasets
    print("** Loading data")
    infile_path = dir_name + '/' + file_name
    test_suffix = ".test" if evid_idx.size==0 else ".cmll"
    train_dset = np.loadtxt(infile_path + ".train", delimiter=',', dtype=int)
    train_seqs = np.loadtxt(infile_path + ".train.seq", delimiter=',', ndmin=1, dtype=int)
    test_dset = np.loadtxt(infile_path + test_suffix, delimiter=',', dtype=int)
    test_seqs = np.loadtxt(infile_path + test_suffix + ".seq", delimiter=',', ndmin=1, dtype=int)
    dom_vals = [vals for vals in np.loadtxt(infile_path + ".doms", skiprows=1, delimiter=",", dtype=int)]
    # Learn model
    print("** Learning Model")
    drcn = DRCN()
    # np.random.seed(1)
    drcn.learn(train_dset, train_seqs, type, file_name, prior_depth, trans_cond_depth,
                trans_cn_depth, b, K, M, maxiter, model_path, dom_vals)
    # print("** Computing Transition LL")
    transition_dset = hmm.create_transition_dset(train_dset, train_seqs)
    # print(drcn.transition_model.avgLL(transition_dset))
    print("** Computing Training LL")
    # print(drcn.avgLL(train_dset, train_seqs))
    if evid_idx.size==0:
        print("** Computing Test LL")
        print(drcn.avgLL(test_dset, test_seqs))
    else:
        print("** Computing Test ELL")
        print(drcn.avgELL(test_dset, test_seqs, evid_idx))
    print("** Finished Program")
