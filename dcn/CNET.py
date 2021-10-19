from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import scipy.sparse as sps
from sklearn.neural_network import MLPClassifier
import math
import time
import sys
import copy
import heapq
import Queue
import dcn.util
import cPickle as pickle
from dcn.Util2 import *

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

'''
Class Chow-Liu Tree.
Members:
    nvariables: Number of variables
    xycounts:
        Sufficient statistics: counts of value assignments to all pairs of variables
        Four dimensional array: first two dimensions are variable indexes
        last two dimensions are value indexes 00,01,10,11
    xcounts:
        Sufficient statistics: counts of value assignments to each variable
        First dimension is variable, second dimension is value index [0][1]
    xyprob:
        xycounts converted to probabilities by normalizing them
    xprob:
        xcounts converted to probabilities by normalizing them
    topo_order:
        Topological ordering over the variables
    parents:
        Parent of each node. Parent[i] gives index of parent of variable indexed by i
        If Parent[i]=-9999 then i is the root node
'''
class CLT:
    def __init__(self, orig=None):
        if orig is None:
            self.nvariables = 0
            self.xycounts = np.ones((1, 1, 2, 2), dtype=int)
            self.xcounts = np.ones((1, 2), dtype=int)
            self.xyprob = np.zeros((1, 1, 2, 2))
            self.xprob = np.zeros((1, 2))
            self.topo_order = []
            self.parents = []
        else:
            self.copy_constructor(orig)
    def copy_constructor(self, orig):
        self.nvariables = orig.nvariables
        self.xycounts = copy.deepcopy(orig.xycounts)
        self.xcounts = copy.deepcopy(orig.xcounts)
        self.xyprob = copy.deepcopy(orig.xyprob)
        self.xprob = copy.deepcopy(orig.xprob)
        self.topo_order = copy.deepcopy(orig.topo_order)
        self.parents = copy.deepcopy(orig.parents)
    ''' Retains CLT structure but re-initializes to uniform distribution '''
    def init_uniform(self):
        self.xycounts = np.ones(self.xycounts.shape,dtype=int)
        self.xcounts = np.ones(self.xcounts.shape,dtype=int)
        self.xyprob = Util2.normalize2d(self.xycounts)
        self.xprob = Util2.normalize1d(self.xcounts)
    ''' Learn the structure of the Chow-Liu Tree using the given dataset '''
    def learnStructure(self, dataset):
        self.nvariables = dataset.shape[1]
        # self.xycounts = Util2.compute_xycounts(dataset) + 1 # laplace correction
        # self.xcounts = Util2.compute_xcounts(dataset) + 2 # laplace correction
        self.xycounts = Util2.compute_xycounts(dataset) + SMOOTH
        self.xcounts = Util2.compute_xcounts(dataset) + (2*SMOOTH)
        self.xyprob = Util2.normalize2d(self.xycounts)
        self.xprob = Util2.normalize1d(self.xcounts)
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util2.compute_edge_weights(self.xycounts, self.xcounts) * (-1.0)
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
    ''' Update the Chow-Liu Tree using weighted samples '''
    def update(self, dataset_, weights=np.array([])):
        # Perform Sampling importance resampling based on weights
        # assume that dataset_.shape[0] equals weights.shape[0] because each example has a weight
        if weights.shape[0]==dataset_.shape[0]:
            norm_weights = Util2.normalize(weights)
            indices = np.argwhere(np.random.multinomial(dataset_.shape[0], norm_weights)).ravel()
            dataset = dataset_[indices, :]
        else:
            dataset=dataset_
        self.xycounts += Util2.compute_xycounts(dataset)
        self.xcounts += Util2.compute_xcounts(dataset)
        self.xyprob = Util2.normalize2d(self.xycounts)
        self.xprob = Util2.normalize1d(self.xcounts)
        edgemat = Util2.compute_edge_weights(self.xycounts, self.xcounts) * (-1.0)
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
    def computeLL(self,dataset):
        prob=0.0
        for i in range(dataset.shape[0]):
            for x in self.topo_order:
                assignx=dataset[i,x]
                # if root sample from marginal
                if self.parents[x] == -9999:
                    prob+=np.log(self.xprob[x][assignx])
                else:
                    # sample from p(x|y)
                    y = self.parents[x]
                    assigny = dataset[i,y]
                    prob+=np.log(self.xyprob[x, y, assignx, assigny] / self.xyprob[y, x, assigny].sum())
        return prob
    def generate_samples(self, numsamples):
        samples = np.zeros((numsamples, self.nvariables), dtype=int)
        for i in range(numsamples):
            for x in self.topo_order:
                # if root sample from marginal
                if self.parents[x] == -9999:
                    samples[i, x] = int(np.random.random() > self.xprob[x, 0])
                else:
                    # sample from p(x|y)
                    y = self.parents[x]
                    assigny = samples[i, y]
                    prob=self.xyprob[x, y, 0, assigny] / (self.xyprob[x, y, 0, assigny] + self.xyprob[x, y, 1, assigny])
                    samples[i, x] = int(np.random.random() > prob)
        return samples
    def __str__(self):
        self_str = "Topo Order: " + str(self.topo_order)
        self_str += "\nP(0) = " + str(self.xprob[0])
        for v in xrange(1,self.topo_order.size):
            curr_id = self.topo_order[v]
            curr_par = self.parents[curr_id]
            self_str += "\nP(" + str(curr_id) + "|" + str(curr_par) + ") =\n" + \
                        str(self.xyprob[curr_id,curr_par] / self.xyprob[curr_par,curr_id].sum(axis=1))
        return self_str
    def expand(self):
        if self.parents.size != self.topo_order.size:
            self.topo_order = np.append(self.topo_order,np.where(self.parents==-9999)[0][1:])
    @staticmethod
    def multiply(p, q):
        # Expand p and q
        p.expand()
        q.expand()
        # Create Copies
        r = CLT()
        r.nvariables = p.nvariables
        r.xycounts = np.copy(p.xycounts)
        r.xcounts = np.copy(p.xcounts)
        r.xyprob = np.copy(p.xyprob)
        r.xprob = np.copy(p.xprob)
        r.topo_order = list(p.topo_order)
        r.parents = list(p.parents)
        q_copy = copy.deepcopy(q)
        # Create Topological Order Map
        topo_order_map = Util2.find_map(p.topo_order,q.topo_order)
        # Swap Order
        q_copy.xyprob = q_copy.xyprob[topo_order_map]
        q_copy.xyprob = q_copy.xyprob[:,topo_order_map]
        q_copy.xprob = q_copy.xprob[topo_order_map]
        # Calculate Counts
        r.xycounts = p.xyprob * q_copy.xyprob
        r.xcounts = p.xprob * q_copy.xprob
        # Normalize to distributions
        r.xyprob = Util2.normalize2d(r.xycounts)
        r.xprob = Util2.normalize1d(r.xcounts)
        # Construct Chow-Liu Tree
        edgemat = Util2.compute_edge_weights(r.xycounts, r.xcounts) * (-1.0)
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert to a Bayesian Network
        r.topo_order, r.parents = depth_first_order(Tree, 0, directed=False)
        # Return CLT
        return r
    '''
        Calculates the probability of evidence (exact)
        evid_idx: Positional Boolean array for each CLT variable
        evid:     Values of evidence
    '''
    def prob_evid(self, evid_idx, evid, debug=False, topo=False):
        # If no evidence, just return 1
        if evid.size == 0:
            return 1.0
        # If the network is an Independent Bayesian Network
        nvars = self.parents.size
        if self.parents.sum() == (-9999 * self.parents.size):
            return self.xprob[np.where(evid_idx==True)[0],evid].prod()
        # Sometimes zero-probability nodes are removed from topo_order
        if self.parents.size != self.topo_order.size:
            self.topo_order = np.append(self.topo_order,np.where(self.parents==-9999)[0][1:])
        # Sort indices in topological order
        if not topo:
            evid_temp = np.full(evid_idx.size,-1)
            evid_temp[evid_idx] = evid
            evid_idx = np.array([ evid_idx[v] for v in self.topo_order ]) # find some way to speed this up
            evid = np.array([ evid_temp[v] for v in self.topo_order ])[evid_idx]  # find some way to speed this up
        # Calculate parent_order
        transdict = dict(zip(self.topo_order,np.arange(nvars)))
        parents = np.array([ self.parents[v] if self.parents[v] != -9999 else v for v in self.topo_order ])
        parents_order = np.array([ transdict.get(p) for p in parents ])
        # Calculate conditional probabilities
        x_y_prob = ( self.xyprob[self.topo_order,parents,:,:] /
                self.xyprob[self.topo_order,parents,:,:].sum(axis=1).reshape(self.topo_order.size,1,2)
                ).transpose(0,2,1).transpose(1,0,2)
        cond_probs = np.nan_to_num(np.hstack((x_y_prob[0],x_y_prob[1])))
        no_parents = np.where(self.parents==-9999)[0]
        cond_probs[no_parents] = np.tile(self.xprob[no_parents],2)
        if debug: print("Checkpoint 1\ncond_probs=\n",cond_probs)
        # Remove evidence from the network
        all_evid = np.vstack((1-evid,evid,1-evid,evid)).T.astype(float)
        if evid_idx[0]: # if the first element is set as evidence
            all_evid[0] = np.array([1-evid[0],evid[0],1-evid[0],evid[0]])
        cond_probs[evid_idx] = all_evid * cond_probs[evid_idx]
        # Modify further for evidence nodes whose parents are also evidence nodes
        evid_dict = dict(zip(self.topo_order[evid_idx],evid))
        parents_evid = np.array([evid_dict.get(p,-9999) for p in parents])
        evid_both_idx = evid_idx * (parents_evid != -9999)
        evid_selector = np.vstack((1-parents_evid[evid_both_idx],
                                    1-parents_evid[evid_both_idx],
                                    parents_evid[evid_both_idx],
                                    parents_evid[evid_both_idx])).T
        cond_probs[evid_both_idx] = cond_probs[evid_both_idx] * evid_selector
        cond_probs[evid_both_idx] = np.tile(np.vstack((cond_probs[evid_both_idx,::2].sum(axis=1),
                                    cond_probs[evid_both_idx,1::2].sum(axis=1))).T,2)
        if debug: print("Checkpoint 2\ncond_probs=",cond_probs)
        # Propagate the message in reverse topological order
        for v in reversed(xrange(1,cond_probs.shape[0])):
            sum_p0, sum_p1 = cond_probs[v,0:2].sum(), cond_probs[v,2:4].sum()
            cond_probs[parents_order[v],::2] *= sum_p0
            cond_probs[parents_order[v],1::2] *= sum_p1
            if debug: print("Checkpoint 3\ncond_probs=\n",cond_probs)
        # Calculate Final P(evid)
        return max(cond_probs[0,0:2].sum(),cond_probs[0,2:4].sum())
    def prob_evid_old(self, evid_idx, evid, debug=False, topo=False):
        # If no evidence, just return 1
        if evid.size == 0:
            return 1.0
        # If the network is an Independent Bayesian Network
        # if self.parents.sum() == (-9999 * self.parents.size):
        #    return self.xprob[np.where(evid_idx==True)[0],evid].prod()
        nvars = self.topo_order.size
        # Sort indices in topological order
        if not topo:
            evid_temp = np.full(evid_idx.size,-1)
            evid_temp[evid_idx] = evid
            evid_idx = np.array([ evid_idx[v] for v in self.topo_order ]) # find some way to speed this up
            evid = np.array([ evid_temp[v] for v in self.topo_order ])[evid_idx]  # find some way to speed this up
        # Calculate parent_order
        transdict = dict(zip(self.topo_order,np.arange(nvars)))
        parents = np.array([ self.parents[v] if self.parents[v] != -9999 else v for v in self.topo_order ])
        parents_order = np.array([ transdict.get(p) for p in parents ])
        # Calculate conditional probabilities
        x_yprob = self.xyprob[ self.topo_order, parents ] / \
                    self.xyprob[ self.topo_order, parents ].sum(axis=1).reshape(nvars,1,2)
        no_parents = np.where(self.topo_order==parents)[0]
        x_yprob[no_parents] = np.transpose(self.xprob[np.repeat(no_parents,2)].reshape(no_parents.shape[0],2,2),(0,2,1))
        if debug: print("Checkpoint 1\ncond_probs=\n",x_yprob)
        # Remove evidence from the network
        x_yprob[evid_idx,1-evid.astype(int),:] = 0.0
        # Modify further for evidence nodes whose parents are also evidence nodes
        evid_dict = dict(zip(self.topo_order[evid_idx],evid))
        parents_evid = np.array([evid_dict.get(p,-9999) for p in parents])
        evid_both_idx = evid_idx * (parents_evid != -9999)
        x_yprob[evid_both_idx,:,1-parents_evid[evid_both_idx].astype(int)] = 0.0
        if debug: print("Checkpoint 2\ncond_probs=",x_yprob)
        # Propagate the message in reverse topological order
        for v in reversed(xrange(1,x_yprob.shape[0])):
            x_yprob[parents_order[v]] = np.einsum('ij,ki->ij',x_yprob[parents_order[v]],x_yprob[v])
            if debug: print("Checkpoint 3\ncond_probs=\n",x_yprob)
        # Calculate Final P(evid)
        return max(x_yprob[0,:,0].sum(),x_yprob[0,:,1].sum())
    '''
            Calculates the conditional probability given evidence
            evid_idx: Positional Boolean array for each CLT evidence
            evid:     Values of evidence
            marg_idx: Positional Boolean array for each CLT marginal variable
            marg:     Values of marginal evidence variables
    '''
    def prob_marg_evid(self, evid_idx, evid, marg_idx, marg):
        # Join marg and evid indices
        all_idx = marg_idx + evid_idx
        all_evid = np.full(all_idx.size,-9999,dtype=int)
        all_evid[marg_idx], all_evid[evid_idx] = marg, evid
        marg_evid = all_evid[all_idx]
        # Check if indices are valid
        if (marg_idx.sum() + evid_idx.sum() != all_idx.sum()) or \
            (marg_idx.sum() != marg.size or evid_idx.sum() != evid.size):
            return -1.0
        # Calculate conditional
        p_all = self.prob_evid(all_idx,all_evid)
        p_evid = self.prob_evid(evid_idx,evid)
        if p_evid == 0.0: return 0.0
        return p_all / p_evid
    def map_old(self):
        # Sometimes zero-probability nodes are removed from topo_order
        nvars = self.parents.size
        if self.parents.size != self.topo_order.size:
            self.topo_order = np.append(self.topo_order,np.where(self.parents==-9999)[0][1:])
        # Calculate parent_order
        transdict = dict(zip(self.topo_order,np.arange(nvars)))
        parents = np.array([ self.parents[v] if self.parents[v] != -9999 else v for v in self.topo_order ])
        parents_order = np.array([ transdict.get(p) for p in parents ])
        # Calculate conditional probabilities
        x_y_prob = ( self.xyprob[self.topo_order,parents,:,:] /
                self.xyprob[self.topo_order,parents,:,:].sum(axis=1).reshape(self.topo_order.size,1,2)
                ).transpose(0,2,1).transpose(1,0,2)
        cond_probs = np.nan_to_num(np.hstack((x_y_prob[0],x_y_prob[1])))
        no_parents = np.where(self.parents==-9999)[0]
        cond_probs[no_parents] = np.tile(self.xprob[no_parents],2)
        # Propagate the message in reverse topological order
        for v in reversed(xrange(1,cond_probs.shape[0])):
            cond_probs[parents_order[v],0] = (cond_probs[parents_order[v],0] * cond_probs[v,0:2]).max()
            cond_probs[parents_order[v],1] = (cond_probs[parents_order[v],1] * cond_probs[v,2:4]).max()
            cond_probs[parents_order[v],2] = (cond_probs[parents_order[v],2] * cond_probs[v,0:2]).max()
            cond_probs[parents_order[v],3] = (cond_probs[parents_order[v],3] * cond_probs[v,2:4]).max()
        # Find MAP assignment
        map_assign = np.zeros(self.topo_order.size,dtype=int)
        map_val = np.max(cond_probs[0])
        for v in xrange(cond_probs.shape[0]):
            node = self.topo_order[v]
            parent = self.parents[node]
            parent_val = 0 if parent==-9999 else map_assign[parent]
            avail_vals = cond_probs[v,(parent_val*2):((parent_val+1)*2)]
            map_assign[node] = np.argmax(avail_vals)
        # Return MAP
        return (map_val,map_assign)
    def map(self, evid_idx=np.array([]), evid=np.array([]), debug=False):
        # Sometimes zero-probability nodes are removed from topo_order
        nvars = self.parents.size
        if self.parents.size != self.topo_order.size:
            self.topo_order = np.append(self.topo_order,np.where(self.parents==-9999)[0][1:])
        # Sort indices in topological order
        if evid.size > 0:
            evid_temp = np.full(evid_idx.size,-1)
            evid_temp[evid_idx] = evid
            evid_idx = np.array([ evid_idx[v] for v in self.topo_order ]) # find some way to speed this up
            evid = np.array([ evid_temp[v] for v in self.topo_order ])[evid_idx]  # find some way to speed this up
        # Calculate parent_order
        transdict = dict(zip(self.topo_order,np.arange(nvars)))
        parents = np.array([ self.parents[v] if self.parents[v] != -9999 else v for v in self.topo_order ])
        parents_order = np.array([ transdict.get(p) for p in parents ])
        # Calculate conditional probabilities
        x_y_prob = ( self.xyprob[self.topo_order,parents,:,:] /
                self.xyprob[self.topo_order,parents,:,:].sum(axis=1).reshape(self.topo_order.size,1,2)
                ).transpose(0,2,1).transpose(1,0,2)
        cond_probs = np.nan_to_num(np.hstack((x_y_prob[0],x_y_prob[1])))
        no_parents = np.where(self.parents==-9999)[0]
        cond_probs[no_parents] = np.tile(self.xprob[no_parents],2)
        # If Evidence exists
        if evid.size > 0:
            # Remove evidence from the network
            all_evid = np.vstack((1-evid,evid,1-evid,evid)).T.astype(float)
            if evid_idx[0]: # if the first element is set as evidence
                all_evid[0] = np.array([1-evid[0],evid[0],1-evid[0],evid[0]])
            cond_probs[evid_idx] = all_evid * cond_probs[evid_idx]
            # Modify further for evidence nodes whose parents are also evidence nodes
            evid_dict = dict(zip(self.topo_order[evid_idx],evid))
            parents_evid = np.array([evid_dict.get(p,-9999) for p in parents])
            evid_both_idx = evid_idx * (parents_evid != -9999)
            evid_selector = np.vstack((1-parents_evid[evid_both_idx],
                                    1-parents_evid[evid_both_idx],
                                    parents_evid[evid_both_idx],
                                    parents_evid[evid_both_idx])).T
            cond_probs[evid_both_idx] = cond_probs[evid_both_idx] * evid_selector
            cond_probs[evid_both_idx] = np.tile(np.vstack((cond_probs[evid_both_idx,::2].sum(axis=1),
                                    cond_probs[evid_both_idx,1::2].sum(axis=1))).T,2)
        if debug: print("Checkpoint 2\ncond_probs=",cond_probs)
        # Propagate the message in reverse topological order
        for v in reversed(xrange(1,cond_probs.shape[0])):
            cond_probs[parents_order[v],0] = (cond_probs[parents_order[v],0] * cond_probs[v,0:2]).max()
            cond_probs[parents_order[v],1] = (cond_probs[parents_order[v],1] * cond_probs[v,2:4]).max()
            cond_probs[parents_order[v],2] = (cond_probs[parents_order[v],2] * cond_probs[v,0:2]).max()
            cond_probs[parents_order[v],3] = (cond_probs[parents_order[v],3] * cond_probs[v,2:4]).max()
            if debug: print("Checkpoint 3\ncond_probs=\n",cond_probs)
        # Find MAP assignment
        map_assign = np.zeros(self.topo_order.size,dtype=int)
        map_val = np.max(cond_probs[0])
        for v in xrange(cond_probs.shape[0]):
            node = self.topo_order[v]
            parent = self.parents[node]
            parent_val = 0 if parent==-9999 else map_assign[parent]
            if evid.size>0 and evid_idx[v]==True:
                map_assign[node] = evid[evid_idx[:v].sum()]
            else:
                avail_vals = cond_probs[v,(parent_val*2):((parent_val+1)*2)]
                map_assign[node] = np.argmax(avail_vals)
        # Return MAP
        return (map_val,map_assign)

    '''
        Reduces the network by eliminating certain variables
        remove: IDs of variables to remove
    '''
    def reduce_network(self,remove):
        # Calculate IDs to be retained
        retain_ids = np.setdiff1d(self.topo_order,remove)
        if retain_ids.size==0: return
        self.xycounts = self.xycounts[(retain_ids),:,:,:][:,(retain_ids),:,:]
        self.xcounts = self.xcounts[(retain_ids),:]
        # Normalize counts
        self.xyprob = Util2.normalize2d(self.xycounts)
        self.xprob = Util2.normalize1d(self.xcounts)
        # Find minimum spanning tree
        edgemat = Util2.compute_edge_weights(self.xycounts, self.xcounts) * (-1.0)
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
    ''' Returns new CLT after reduction '''
    def reduce_network_new(self,remove):
        new_clt = copy.deepcopy(self)
        new_clt.reduce_network(remove)
        return new_clt
    ''' Returns a CLT with slight structural perturbation by randomly swapping k edges '''
    def perturb(self, K, seed=0):
        # Create a copy of the current CLT
        clt = copy.deepcopy(self)
        # Manually set the seed if it is not zero
        if seed!=0: np.random.seed(seed)
        # Set the edge weights of K random edges in the CLT to 0
        i_idx = np.random.choice(len(clt.topo_order), min(K, len(clt.topo_order)), False)
        j_idx = np.array([ i if clt.parents[i]==-9999 else clt.parents[i] for i in i_idx ])
        edgemat = Util2.compute_edge_weights(clt.xycounts, clt.xcounts) * (-1.0)
        edgemat[i_idx,j_idx] = 0.0
        # Find a new spanning tree and re-learn structure
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        clt.topo_order, clt.parents = depth_first_order(Tree, 0, directed=False)
        # Return perturbed CLT
        return clt

# Code copied from Tahrima.
# Inefficient and will need to speed up one day
class CNET:
    def __init__(self,depth=100):
        self.nvariables=0
        self.depth=depth
        self.tree=[]
        # Chiro
        self.map_assign=None
        self.map_val=None
    def __str__helper(self, tree, level, prob, symbol):
        node = tree
        level_str = symbol * level * 2
        prob_str = '{:03.2f}'.format(prob)
        if type(node) != type([]):
            return level_str + " " + prob_str + ": " + \
                    "CLT Topo: " + str(node.topo_order) + "\n"
        left_str = self.__str__helper(node[4], level+1, node[2]/sum(node[2:4]), '-')
        right_str = self.__str__helper(node[5], level+1, node[3]/sum(node[2:4]), '+')
        return level_str + " " + prob_str + ": " + str(node[1]) + "\n" + \
                left_str + right_str
    ''' String representation of cutset network '''
    def __str__(self):
        return self.__str__helper(self.tree,0,1.0,'')
    def save(self,dir,filename="cnet"):
        outfilename = dir + '/' + filename + ".model"
        with open(outfilename,'wb') as outfile:
            pickle.dump(self,outfile,pickle.HIGHEST_PROTOCOL)
    def load(self,dir,filename="cnet"):
        infilename = dir + '/' + filename + ".model"
        with open(infilename,'rb') as infile:
            obj = pickle.load(infile)
            self.nvariables = obj.nvariables
            self.depth = obj.depth
            self.tree = obj.tree
            self.map_assign = obj.map_assign
            self.map_val = obj.map_val
    def learnStructureHelper(self,dataset,ids):
        curr_depth=self.nvariables-dataset.shape[1]
        if dataset.shape[0]<10 or dataset.shape[1]<5 or curr_depth >= self.depth:
            clt=CLT()
            clt.learnStructure(dataset)
            return clt
        xycounts = Util2.compute_xycounts(dataset) + 1  # laplace correction
        xcounts = Util2.compute_xcounts(dataset) + 2  # laplace correction
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util2.compute_edge_weights(xycounts, xcounts)
        scores = np.sum(edgemat, axis=0)
        #print (scores)
        variable = np.argmax(scores)
        new_dataset1=np.delete(dataset[dataset[:,variable]==1],variable,1)
        p1=float(new_dataset1.shape[0])+1.0
        new_ids=np.delete(ids,variable,0)
        new_dataset0 = np.delete(dataset[dataset[:, variable] == 0], variable, 1)
        p0 = float(new_dataset0.shape[0]) +1.0
        return [variable,ids[variable],p0,p1,self.learnStructureHelper(new_dataset0,new_ids),
                self.learnStructureHelper(new_dataset1,new_ids)]
    def learnStructure(self, dataset):
        self.nvariables = dataset.shape[1]
        ids=np.arange(self.nvariables)
        self.tree=self.learnStructureHelper(dataset,ids)
        #print(self.tree)
    def computeLL(self,dataset,vec_flag=False):
        probs = np.zeros(dataset.shape[0])
        for i in range(dataset.shape[0]):
            evid_idx = np.array([True]*dataset.shape[1])
            evid = dataset[i,:]
            probs[i] = np.log(self.prob_evid(self.tree,evid_idx,evid))
        if vec_flag:
            return probs
        return probs.sum()
    def update(self,dataset_, weights=np.array([])):
        if weights.shape[0]==dataset_.shape[0]:
            norm_weights = Util2.normalize(weights)
            indices = np.argwhere(np.random.multinomial(dataset_.shape[0], norm_weights)).ravel()
            dataset = dataset_[indices, :]
        else:
            dataset=dataset_
        for i in range(dataset.shape[0]):
            node=self.tree
            ids=np.arange(self.nvariables)
            while isinstance(node,list):
                id,x,p0,p1,node0,node1=node
                p0_index=2
                p1_index=3
                assignx=dataset[i,x]
                ids=np.delete(ids,id,0)
                if assignx==1:
                    node[p1_index]=p1+1.0
                    node=node1
                else:
                    node[p0_index]=p0+1.0
                    node = node0
            node.update(dataset[i:i+1,ids])
    @staticmethod
    def is_same_structure(p,q,id=True):
        active = Queue.Queue()
        active.put((p.tree,q.tree))
        while not active.empty():
            p_curr, q_curr = active.get()
            if type(p_curr) != type(q_curr):
                # print('*** Point of difference found!')
                # print(str(p_curr))
                # print(str(q_curr))
                return False
            if type(p_curr) != type([]):
                continue
            if id==True and p_curr[1] != q_curr[1]:
                return False
            active.put((p_curr[4],q_curr[4]))
            active.put((p_curr[5],q_curr[5]))
        return True
    def prob_evid(self, tree, evid_idx, evid, assign_idx=np.array([]), debug=False):
        node = tree
        if evid.size == 0:
            return 1.0
        if assign_idx.size != evid_idx.size:
            assign_idx = np.array([False]*evid_idx.size)
        if type(node) != type([]):
            clt_evid_idx = evid_idx[np.invert(assign_idx)]
            return node.prob_evid(clt_evid_idx,evid,debug)
        id = node[1]
        evid_id = evid_idx[:id].sum()
        assign_idx[id] = True
        p_left = node[2]/sum(node[2:4])
        p_right = node[3]/sum(node[2:4])
        p_left_tree, p_right_tree = 0.0, 0.0
        # If current node is evidence, then modify evid_idx and evid
        if evid_idx[id]:
            evid_idx[id] = False
            evid_new = np.array(np.hstack((evid[:evid_id],evid[evid_id+1:])))
            if evid[evid_id]==0:
                p_left_tree = self.prob_evid(node[4], evid_idx, evid_new, assign_idx, debug)
            else:
                p_right_tree = self.prob_evid(node[5], evid_idx, evid_new, assign_idx, debug)
            evid_idx[id] = True
        # If current node is not evidence, then just call recursively
        else:
            p_left_tree = self.prob_evid(node[4], evid_idx, evid, assign_idx, debug)
            p_right_tree = self.prob_evid(node[5], evid_idx, evid, assign_idx, debug)
        assign_idx[id] = False
        return (p_left * p_left_tree) + (p_right * p_right_tree)
    def prob_marg_evid(self, evid_idx, evid, marg_idx, marg):
        # Join marg and evid indices
        all_idx = marg_idx + evid_idx
        all_evid = np.full(all_idx.size,-9999,dtype=int)
        all_evid[marg_idx], all_evid[evid_idx] = marg, evid
        marg_evid = all_evid[all_idx]
        # Check if indices are valid
        if (marg_idx.sum() + evid_idx.sum() != all_idx.sum()) or \
            (marg_idx.sum() != marg.size or evid_idx.sum() != evid.size):
            return -1.0
        # Calculate conditional
        p_all = self.prob_evid(self.tree,all_idx,marg_evid)
        p_evid = self.prob_evid(self.tree,evid_idx,evid)
        if p_evid == 0.0: return 0.0
        return p_all / p_evid
    def kmap_helper(self,node,particles,curr_map_assign,curr_map_val,lvl,evid_idx,evid,mul=1.0,num_vars=999999999):
        if not isinstance(node,list):
            # Adjust evidence Array
            if evid.size > 0:
                assign_idx = (curr_map_assign+1).astype(bool)
                evid_idx_idx = evid_idx.copy().astype(int)
                evid_idx_idx[evid_idx] = np.arange(evid_idx.sum())
                evid = evid[evid_idx_idx[evid_idx * (1-assign_idx).astype(bool)]]
                evid_idx = evid_idx[(1-assign_idx).astype(bool)]
            # Calculate MAP for leaf node
            leaf_map_val, leaf_map_assign = node.map(evid_idx,evid)
            leaf_map_val = math.log(leaf_map_val) if leaf_map_val!=0 else -9999
            # Merge current MAP values with leaf MAP values
            self.map_val, self.map_assign = curr_map_val, np.copy(curr_map_assign)
            Util2.merge(self.map_assign,leaf_map_assign,-1)
            self.map_val += leaf_map_val
            # Push particle into priority queue
            pid = len(particles) + 1
            # if(self.map_assign.sum()>4):
            #    self.map_val = -9999
            heapq.heappush(particles,(-self.map_val, mul*pid, self.map_assign[:num_vars]))
        else:
            # Unpack node information
            name, id, p0, p1, node0, node1 = node
            # Set current variable to 0 and traverse the left subtree
            curr_map_assign[id] = 0
            curr_map_val += math.log(p0/(p0+p1))
            if evid.size==0 or not evid_idx[id] or (evid_idx[id] and evid[evid_idx[:id].sum()]==0):
                self.kmap_helper(node0,particles,curr_map_assign,curr_map_val,lvl+1,evid_idx,evid,mul,num_vars)
            # Set current variable to 1 and traverse the right subtree
            curr_map_assign[id] = 1
            curr_map_val += (math.log(p1/(p0+p1))-math.log(p0/(p0+p1)))
            if evid.size==0 or not evid_idx[id] or (evid_idx[id] and evid[evid_idx[:id].sum()]==1):
                self.kmap_helper(node1,particles,curr_map_assign,curr_map_val,lvl+1,evid_idx,evid,mul,num_vars)
            # Reset values of curr_map_assign and curr_map_val
            curr_map_assign[id] = -1
            curr_map_val -= math.log(p1/(p0+p1))
    def kmap(self,k=1,evid_idx=np.array([]),evid=np.array([]),mul=1.0,num_vars=999999999,log_flag=False):
        # Initialize variables
        node = self.tree
        particles = []
        curr_map_val, self.map_val = 0.00, -np.inf
        curr_map_assign, self.map_assign = np.full(self.nvariables,-1,dtype=int), np.full(self.nvariables,-1,dtype=int)
        # Get particles
        self.kmap_helper(node,particles,curr_map_assign,curr_map_val,1,evid_idx,evid,mul,num_vars)
        # Return the top-k particles
        k_particles = []
        for p in xrange(k):
            if len(particles)==0:
                break
            particle = heapq.heappop(particles)
            score = particle[0] if log_flag else math.exp(-particle[0])
            k_particles.append((score,particle[1],particle[2]))
        return k_particles
    def map(self, evid_idx=np.array([]), evid=np.array([])):
        map_particle = self.kmap(1,evid_idx,evid)[0]
        return (map_particle[0], map_particle[2])
    def generate_samples(self,numsamples=100,assign_idx=np.array([]),assign=np.array([]),outdir="",outfilename="cnet",sort_samples=False,seed_id=0):
        # Assertions
        assign_idx = np.array([False]*self.nvariables) if assign_idx.size==0 else assign_idx
        if numsamples <= 0 or assign_idx.size != self.nvariables or assign.size != assign_idx.sum():
            return np.array([])
        # Browse to assignment
        curr_node = self.tree
        for i in xrange(assign.size):
            if type(curr_node)!=type([]) or not assign_idx[curr_node[1]]:
                return np.array([])
            id = curr_node[1]
            a = assign_idx[:id].sum()
            curr_node = curr_node[4+assign[a]]
        # Sample
        samples = np.full((numsamples,self.nvariables),-1,dtype=int)
        samples[:,assign_idx] = assign
        root_node = curr_node
        sorted_samples = []
        for m in xrange(numsamples):
            curr_node = root_node
            while type(curr_node) == type([]):
                _, id, p0, p1, left, right = curr_node
                u = np.random.uniform()
                if u <= p0/(p0+p1):
                    samples[m,id] = 0
                    curr_node = left
                else:
                    samples[m,id] = 1
                    curr_node = right
            if (samples[m,:]==-1).sum() > 0:
                clt_sample = curr_node.generate_samples(1)[0]
                samples[m,samples[m,:]==-1] = clt_sample
                if sort_samples:
                    sorted_samples.append((np.exp(self.computeLL(samples[m:m+1,:])),seed_id+m,samples[m]))
        # Write to file
        if len(outdir)>0:
            filename = outdir + outfilename + ".samples.txt"
            np.savetxt(filename,samples,fmt='%i',delimiter=',')
        # Return
        if sort_samples:
            return sorted(sorted_samples)
        return samples
    def weighted_samples(self, numsamples=100, evid_idx=np.array([]), evid=np.array([]), mul=1.0):
        # Validation
        if (numsamples <= 0 or evid_idx.sum() != evid.size):
            return np.array([])
        # Check if no evidence
        initial_samples = self.generate_samples(numsamples)
        if (evid_idx.size==0):
            # evid_idx = np.array([ False for v in range(self.nvariables) ])
            return initial_samples
        # Otherwise
        weighted_samples = []
        non_evid_idx = ~evid_idx
        # evid_weight = self.prob_evid(self.tree, evid_idx, evid)
        for s in range(len(initial_samples)):
            curr_sample = initial_samples[s]
            curr_sample[evid_idx] = evid
            # non_evid = curr_sample[non_evid_idx]
            # weight = -np.log(self.prob_marg_evid(evid_idx, evid, non_evid_idx, non_evid))
            weight = self.prob_evid(self.tree, np.array([True]*self.nvariables), curr_sample)
            # weight = -np.log(weight) if (weight == 0.0) else -np.log(weight * mul)
            weighted_samples += [ (weight*mul, curr_sample) ]
        return sorted(weighted_samples, key=lambda x: -x[0])
    '''
        For the distribution P(Xt+1,Xt), calculates reduced distribution
        P(Xt+1,~Xt) where all the Xt variables from the leaf CLTs are
        eliminated
    '''
    def reduce_dist_helper(self, tree, count, total):
        node = tree
        if type(node) != type([]):
            remove = np.arange(total-count)
            if remove.size>0: node.reduce_network(remove)
        else:
            if node[1] >= total:
                count = count+1
            self.reduce_dist_helper(node[4], count, total)
            self.reduce_dist_helper(node[5], count, total)
            count = count - 1
    def reduce_dist(self, total):
        self.reduce_dist_helper(self.tree,0,total)
    ''' Finds the shallowest sub-tree rooted at a variable with var.id < id '''
    def find_sub_tree(self, id):
        queue = Queue.Queue()
        new_tree = None
        queue.put(self.tree)
        while not new_tree and not queue.empty():
            node = queue.get()
            if type(node) != type([]):
                continue
            queue.put(node[4])
            queue.put(node[5])
            if node[1] < id:
                new_tree = node
        return new_tree
    '''
        Given a reduced distribution P(Xt+1,~Xt), learn a compatible
        structure Q that can be used for dynamic projection updates
    '''
    def learn_project_structure_helper(self, tree, total):
        node = tree
        if type(node) != type([]):
            clt = copy.deepcopy(node)
            clt.init_uniform()
            return clt
        if node[1] < total:
            left_tree = self.learn_project_structure_helper(node[4], total)
            right_tree = self.learn_project_structure_helper(node[5], total)
            return [ node[0], node[1], 1.0, 1.0, left_tree, right_tree ]
        else:
            if node[2] > node[3]:
                return self.learn_project_structure_helper(node[4], total)
            else:
                return self.learn_project_structure_helper(node[5], total)
    def learn_project_structure(self, total):
        root_tree = self.find_sub_tree(total)
        left_tree = self.learn_project_structure_helper(root_tree[4], total)
        right_tree = self.learn_project_structure_helper(root_tree[5], total)
        q = CNET(self.depth)
        q.nvariables = total
        q.tree = [ root_tree[0], root_tree[1], 1.0, 1.0, left_tree, right_tree ]
        return q
    '''
        Obtain CLT R(X|Y) from two conditional distributions P(X|Y) and Q(X|Y)
        defined on exactly the same variables (slow)
    '''
    @staticmethod
    def multiply_into_clt(p, q, assign_idx, assign):
        r = CLT()
        nvars = (1-assign_idx).sum()
        # Check if everything is assigned
        if nvars==0:
            return r
        # Initialize Counts, Marginals and Temps
        r.nvariables = nvars
        r.xycounts = np.ones((nvars,nvars,2,2),dtype=float)
        r.xcounts = np.ones((nvars,2),dtype=float)
        marg_idx = np.array([False]*assign_idx.size)
        marg_vars = np.where(assign_idx==False)[0]
        a00, a01, a10, a11 = np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])
        # Generate counts
        for x in xrange(nvars):
            marg_idx[marg_vars[x]] = True
            r.xcounts[x,0] = p.prob_marg_evid(assign_idx,assign,marg_idx,a01[0]) * \
                            q.prob_marg_evid(assign_idx,assign,marg_idx,a01[0])
            r.xcounts[x,1] = p.prob_marg_evid(assign_idx,assign,marg_idx,a01[1]) * \
                            q.prob_marg_evid(assign_idx,assign,marg_idx,a01[1])
            for y in xrange(x,nvars):
                if x==y:
                    r.xycounts[x,y,0,0], r.xycounts[x,y,1,1] = r.xcounts[x,0], r.xcounts[x,1]
                    r.xycounts[x,y,0,1], r.xycounts[x,y,1,0] = r.xcounts[x,0] * r.xcounts[x,1], \
                                                                r.xcounts[x,0] * r.xcounts[x,1]
                    continue
                marg_idx[marg_vars[y]] = True
                r.xycounts[x,y,0,0] = p.prob_marg_evid(assign_idx,assign,marg_idx,a00) * \
                                        q.prob_marg_evid(assign_idx,assign,marg_idx,a00)
                r.xycounts[x,y,0,1] = p.prob_marg_evid(assign_idx,assign,marg_idx,a01) * \
                                        q.prob_marg_evid(assign_idx,assign,marg_idx,a01)
                r.xycounts[x,y,1,0] = p.prob_marg_evid(assign_idx,assign,marg_idx,a10) * \
                                        q.prob_marg_evid(assign_idx,assign,marg_idx,a10)
                r.xycounts[x,y,1,1] = p.prob_marg_evid(assign_idx,assign,marg_idx,a11) * \
                                        q.prob_marg_evid(assign_idx,assign,marg_idx,a11)
                r.xycounts[y,x] = r.xycounts[x,y].T
                marg_idx[marg_vars[y]] = False
            marg_idx[marg_vars[x]] = False
        # Normalize and Generate Spanning Tree
        r.xyprob = Util2.normalize2d(r.xycounts)
        r.xprob = Util2.normalize1d(r.xcounts)
        edgemat = Util2.compute_edge_weights(r.xycounts, r.xcounts) * (-1.0)
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        r.topo_order, r.parents = depth_first_order(Tree, 0, directed=False)
        return r
    @staticmethod
    def multiply_into_clt2(clt, p, assign_idx, assign, retain=False, prior=False):
        # Assertions
        if not isinstance(clt,CLT) or clt.xprob.shape[0] > p.nvariables or \
            assign_idx.size != p.nvariables or assign_idx.sum() != assign.size:
            return CLT(clt)
        # Initializations
        r = CLT(clt)
        nvars = clt.xprob.shape[0]
        totvars = assign_idx.size
        r.xycounts = np.ones((nvars,nvars,2,2),dtype=float)
        r.xcounts = np.ones((nvars,2),dtype=float)
        # marg_idx = np.array([False]*assign_idx.size)
        marg_idx = np.hstack((np.array([False]*totvars),np.array([False]*totvars*(not prior),dtype=bool)))
        marg_vars = np.where(assign_idx==False)[0]
        evid_idx = np.copy(marg_idx)
        evid = np.copy(assign)
        evid_idx[:assign_idx.size] = assign_idx
        a00, a01, a10, a11 = np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])
        # Generate counts
        for x in xrange(nvars):
            marg_idx[marg_vars[x]] = True
            r.xcounts[x,0] = clt.xcounts[x,0] * p.prob_marg_evid(evid_idx,evid,marg_idx,a01[0])
            r.xcounts[x,1] = clt.xcounts[x,1] * p.prob_marg_evid(evid_idx,evid,marg_idx,a01[1])
            if retain:
                y_arr = np.array([clt.parents[x]]) if clt.parents[x] != -9999 else np.array([x])
            else:
                y_arr = np.arange(x,nvars)
            for y in y_arr:
                if x==y:
                    r.xycounts[x,y,0,0], r.xycounts[x,y,1,1] = r.xcounts[x,0], r.xcounts[x,1]
                    r.xycounts[x,y,0,1], r.xycounts[x,y,1,0] = 0.0, 0.0
                    continue
                marg_idx[marg_vars[y]] = True
                p00 = p.prob_marg_evid(evid_idx,evid,marg_idx,a00)
                p01 = p.prob_marg_evid(evid_idx,evid,marg_idx,a01)
                p10 = p.prob_marg_evid(evid_idx,evid,marg_idx,a10)
                p11 = 1 - (p00+p01+p10)
                r.xycounts[x,y,0,0] = clt.xycounts[x,y,0,0] * p00
                r.xycounts[x,y,0,1] = clt.xycounts[x,y,0,1] * p01
                r.xycounts[x,y,1,0] = clt.xycounts[x,y,1,0] * p10
                r.xycounts[x,y,1,1] = clt.xycounts[x,y,1,1] * p11
                r.xycounts[y,x] = r.xycounts[x,y].T
                marg_idx[marg_vars[y]] = False
            marg_idx[marg_vars[x]] = False
        # Normalize and Generate Spanning Tree
        r.xyprob = Util2.normalize2d(r.xycounts)
        r.xprob = Util2.normalize1d(r.xcounts)
        if not retain:
            edgemat = Util2.compute_edge_weights(r.xycounts, r.xcounts) * (-1.0)
            Tree = minimum_spanning_tree(csr_matrix(edgemat))
            r.topo_order, r.parents = depth_first_order(Tree, 0, directed=False)
        return r
    @staticmethod
    def multiply_into_clt4(p, q, assign_idx, assign, retain=False, q_clt=None):
        # Assertions
        if assign_idx.size != q.nvariables or assign_idx.sum() != assign.size:
            return CLT()
        # Initialize CLT
        r = CLT()
        nvars = (1-assign_idx).sum()
        totvars = assign_idx.size
        r.nvariables = nvars
        r.xycounts = np.ones((nvars,nvars,2,2),dtype=float)
        r.xcounts = np.ones((nvars,2),dtype=float)
        # Initialize CLT Update Node
        q_id = 0 if isinstance(q.tree,CLT) else q.tree[1]
        evid_idx_q = np.array(([False]*q_id)+[True]+([False]*(totvars-q_id-1)))
        q0 = q.prob_evid(q.tree,evid_idx_q,np.array([0]))
        q1 = 1-q0
        # Initialize Query Parameters
        marg_idx = np.array([False]*totvars*2)
        marg_vars = np.where(assign_idx==False)[0]
        evid_idx = np.hstack((assign_idx,np.array([False]*totvars)))
        evid_idx[totvars+q_id] = True
        evid = np.copy(assign)
        a00, a01, a10, a11 = np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])
        # Generate counts
        for x in xrange(nvars):
            marg_idx[marg_vars[x]] = True
            # evid_idx[totvars+marg_vars[x]] = True
            evid0 = np.hstack((evid,np.array([0])))
            evid1 = np.hstack((evid,np.array([1])))
            # q0 = q.prob_evid(q.tree,marg_idx[:totvars],np.array([0]))
            # q1 = 1 - q0
            p0_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,a01[0])
            p0_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,a01[0])
            p1_0 = 1 - p0_0
            p1_1 = 1 - p0_1
            r.xcounts[x,0] = (q0 * p0_0) + (q1 * p0_1)
            r.xcounts[x,1] = (q0 * p1_0) + (q1 * p1_1)
            if retain and isinstance(q_clt,CLT):
                y_arr = np.array([q_clt.parents[x]]) if q_clt.parents[x] != -9999 else np.array([x])
            else:
                y_arr = np.arange(x,nvars)
            for y in y_arr:
                if x==y:
                    r.xycounts[x,y,0,0], r.xycounts[x,y,1,1] = r.xcounts[x,0], r.xcounts[x,1]
                    r.xycounts[x,y,0,1], r.xycounts[x,y,1,0] = r.xcounts[x,0] * r.xcounts[x,1], \
                                                                    r.xcounts[x,0] * r.xcounts[x,1]
                    continue
                marg_idx[marg_vars[y]] = True
                # Calculate P's
                p00_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,a00)
                p01_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,a01)
                p10_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,a10)
                p11_0 = 1 - (p00_0+p01_0+p10_0)
                p00_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,a00)
                p01_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,a01)
                p10_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,a10)
                p11_1 = 1 - (p00_1+p01_1+p10_1)
                r.xycounts[x,y,0,0] = (p00_0*q0) + (p00_1*q1)
                r.xycounts[x,y,0,1] = (p01_0*q0) + (p01_1*q1)
                r.xycounts[x,y,1,0] = (p10_0*q0) + (p10_1*q1)
                r.xycounts[x,y,1,1] = (p11_0*q0) + (p11_1*q1)
                r.xycounts[y,x] = r.xycounts[x,y].T
                marg_idx[marg_vars[y]] = False
            marg_idx[marg_vars[x]] = False
        # Normalize and Generate Spanning Tree
        r.xyprob = Util2.normalize2d(r.xycounts)
        r.xprob = Util2.normalize1d(r.xcounts)
        if not retain:
            edgemat = Util2.compute_edge_weights(r.xycounts, r.xcounts) * (-1.0)
            Tree = minimum_spanning_tree(csr_matrix(edgemat))
            r.topo_order, r.parents = depth_first_order(Tree, 0, directed=False)
        elif retain and isinstance(q_clt,CLT):
            r.topo_order, r.parents = q_clt.topo_order.copy(), q_clt.parents.copy()
        return r
    def multiply_into_clt4_smooth(p, q, assign_idx, assign, retain=False, q_clt=None):
        # print('[DEBUG] Entering multiply_into_clt4')
        # Assertions
        if assign_idx.size != q.nvariables or assign_idx.sum() != assign.size:
            return CLT()
        # Initialize CLT
        r = CLT()
        nvars = (1-assign_idx).sum()
        totvars = assign_idx.size
        r.nvariables = nvars
        r.xycounts = np.ones((nvars,nvars,2,2),dtype=float)
        r.xcounts = np.ones((nvars,2),dtype=float)
        # Initialize CLT Update Node
        q_id = 0 if isinstance(q.tree,CLT) else q.tree[1]
        evid_idx_q = np.array(([False]*q_id)+[True]+([False]*(totvars-q_id-1)))
        q0 = q.prob_evid(q.tree,evid_idx_q,np.array([0]))
        q1 = 1-q0
        # Initialize Query Parameters
        marg_idx = np.array([False]*totvars*2)
        marg_vars = np.where(assign_idx==False)[0]
        evid_idx = np.hstack((np.array([False]*totvars),assign_idx))
        evid_idx[q_id] = True
        evid = np.copy(assign)
        a00, a01, a10, a11 = np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])
        # Generate counts
        for x in xrange(nvars):
            marg_idx[totvars+marg_vars[x]] = True
            # evid_idx[totvars+marg_vars[x]] = True
            evid0 = np.hstack((np.array([0]),evid))
            evid1 = np.hstack((np.array([1]),evid))
            # q0 = q.prob_evid(q.tree,marg_idx[:totvars],np.array([0]))
            # q1 = 1 - q0
            p0_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,a01[0])
            p0_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,a01[0])
            p1_0 = 1 - p0_0
            p1_1 = 1 - p0_1
            r.xcounts[x,0] = (q0 * p0_0) + (q1 * p0_1)
            r.xcounts[x,1] = (q0 * p1_0) + (q1 * p1_1)
            if retain and isinstance(q_clt,CLT):
                y_arr = np.array([q_clt.parents[x]]) if q_clt.parents[x] != -9999 else np.array([x])
            else:
                y_arr = np.arange(x,nvars)
            for y in y_arr:
                if x==y:
                    r.xycounts[x,y,0,0], r.xycounts[x,y,1,1] = r.xcounts[x,0], r.xcounts[x,1]
                    r.xycounts[x,y,0,1], r.xycounts[x,y,1,0] = r.xcounts[x,0] * r.xcounts[x,1], \
                                                                    r.xcounts[x,0] * r.xcounts[x,1]
                    continue
                marg_idx[totvars+marg_vars[y]] = True
                # evid_idx[totvars+marg_vars[y]] = True
                # evid00, evid01 = np.hstack((assign,a00)), np.hstack((assign,a01))
                # evid10, evid11 = np.hstack((assign,a10)), np.hstack((assign,a11))
                # Calculate Q's
                # q00 = q.prob_evid(q.tree,marg_idx[:totvars],a00)
                # q01 = q.prob_evid(q.tree,marg_idx[:totvars],a01)
                # q10 = q.prob_evid(q.tree,marg_idx[:totvars],a10)
                # q11 = 1 - (q00+q01+q10)
                # Calculate P's
                # p00_00 = p.prob_marg_evid(evid_idx,evid00,marg_idx,a00)
                # p00_01 = p.prob_marg_evid(evid_idx,evid01,marg_idx,a00)
                # p00_10 = p.prob_marg_evid(evid_idx,evid10,marg_idx,a00)
                # p00_11 = 1 - (p00_00+p00_01+p00_10)
                # p01_00 = p.prob_marg_evid(evid_idx,evid00,marg_idx,a01)
                # p01_01 = p.prob_marg_evid(evid_idx,evid01,marg_idx,a01)
                # p01_10 = p.prob_marg_evid(evid_idx,evid10,marg_idx,a01)
                # p01_11 = 1 - (p01_00+p01_01+p01_10)
                # p10_00 = p.prob_marg_evid(evid_idx,evid00,marg_idx,a10)
                # p10_01 = p.prob_marg_evid(evid_idx,evid01,marg_idx,a10)
                # p10_10 = p.prob_marg_evid(evid_idx,evid10,marg_idx,a10)
                # p10_11 = 1 - (p10_00+p10_01+p10_10)
                # p11_00 = p.prob_marg_evid(evid_idx,evid00,marg_idx,a11)
                # p11_01 = p.prob_marg_evid(evid_idx,evid01,marg_idx,a11)
                # p11_10 = p.prob_marg_evid(evid_idx,evid10,marg_idx,a11)
                # p11_11 = 1 - (p11_00+p11_01+p11_10)
                # r.xycounts[x,y,0,0] = (p00_00*q00) + (p00_01*q01) + (p00_10*q10) + (p00_11*q11)
                # r.xycounts[x,y,0,1] = (p01_00*q00) + (p01_01*q01) + (p01_10*q10) + (p01_11*q11)
                # r.xycounts[x,y,1,0] = (p10_00*q00) + (p10_01*q01) + (p10_10*q10) + (p10_11*q11)
                # r.xycounts[x,y,1,1] = (p11_00*q00) + (p11_01*q01) + (p11_10*q10) + (p11_11*q11)
                # Calculate P's
                p00_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,a00)
                p01_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,a01)
                p10_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,a10)
                p11_0 = 1 - (p00_0+p01_0+p10_0)
                p00_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,a00)
                p01_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,a01)
                p10_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,a10)
                p11_1 = 1 - (p00_1+p01_1+p10_1)
                r.xycounts[x,y,0,0] = (p00_0*q0) + (p00_1*q1)
                r.xycounts[x,y,0,1] = (p01_0*q0) + (p01_1*q1)
                r.xycounts[x,y,1,0] = (p10_0*q0) + (p10_1*q1)
                r.xycounts[x,y,1,1] = (p11_0*q0) + (p11_1*q1)
                r.xycounts[y,x] = r.xycounts[x,y].T
                marg_idx[totvars+marg_vars[y]] = False
                # evid_idx[totvars+marg_vars[y]] = False
            marg_idx[totvars+marg_vars[x]] = False
            # evid_idx[totvars+marg_vars[x]] = False
        # Normalize and Generate Spanning Tree
        r.xyprob = Util2.normalize2d(r.xycounts)
        r.xprob = Util2.normalize1d(r.xcounts)
        if not retain:
            edgemat = Util2.compute_edge_weights(r.xycounts, r.xcounts) * (-1.0)
            Tree = minimum_spanning_tree(csr_matrix(edgemat))
            r.topo_order, r.parents = depth_first_order(Tree, 0, directed=False)
        elif retain and isinstance(q_clt,CLT):
            r.topo_order, r.parents = q_clt.topo_order, q_clt.parents
        # print('[DEBUG] Exiting multiply_into_clt4')
        return r
    '''
        Given a rooted tree and a set of assignments to Xt+1, fetch the
        corresponding CLT
    '''
    @staticmethod
    def fetch_clt(tree,assign_idx,assign):
        clt = tree
        total = assign_idx.size
        for i in xrange(assign.size):
            while clt[1]>=total:
                clt = clt[4] # if clt[2]>clt[3] else clt[5]
            id = clt[1]
            # print('id: '+str(id))
            a = assign_idx[:id].sum()
            clt = clt[4+assign[a]]
        # In case there are still some >=id nodes left
        while type(clt) == type([]):
            clt = clt[4] # if clt[2]>clt[3] else clt[5]
        return clt
    '''
        Project a prior distribution P(Xt|et) or a reduced distribution
        P(Xt+1,~Xt|et+1) onto Q(Xt)
    '''
    def project_and_update_helper(self, q_tree, p, q_old, assign_idx, assign, retain, prior):
        # Initialization
        node = q_tree
        id = node[1]
        marg_idx = np.array([False]*2*self.nvariables)
        evid_idx = np.hstack((assign_idx,[False]*self.nvariables))
        # Set parameters
        marg_idx[id] = True
        marg0, marg1 = np.array([0]), np.array([1])
        a = assign_idx[:id].sum()
        assign_idx_new = np.hstack((assign_idx[:id],np.array([True]),assign_idx[id+1:]))
        assign_new0 = np.hstack((assign[:a],marg0,assign[a:]))
        assign_new1 = np.hstack((assign[:a],marg1,assign[a:]))
        evid_idx[id+self.nvariables] = True
        evid0 = np.hstack((assign, marg0))
        evid1 = np.hstack((assign, marg1))
        # Adjust parameters for prior update
        if prior:
            marg_idx = marg_idx[:self.nvariables]
            evid_idx = evid_idx[:self.nvariables]
            evid0, evid1 = evid0[:-1], evid1[:-1]
        # Perform inference on current and transition distributions
        q0 = q_old.prob_evid(q_old.tree,marg_idx[:self.nvariables],marg0)
        q1 = 1-q0
        p0_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,marg0)
        p0_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,marg0)
        p1_0 = 1-p0_0
        p1_1 = 1-p0_1
        # Update branch parameters for Q(Xt_pid=0|~Xt_asgn)
        node[2] = (q0*p0_0) + (q1*p0_1)
        # If left branch is CLT update, otherwise update left sub-tree
        if type(node[4]) != type([]):
            if prior:
                node[4] = CNET.multiply_into_clt2(node[4], p, assign_idx_new, assign_new0, retain, prior)
            else:
                node[4] = CNET.multiply_into_clt4(p, q_old, assign_idx_new, assign_new0, retain, node[4])
        else:
            self.project_and_update_helper(q_tree[4], p, q_old, assign_idx_new, assign_new0, retain, prior)
        # Update branch parameters for Q(Xt_pid=1|~Xt_asgn)
        node[3] = 1 - node[2]
        # If right branch is CLT update, otherwise update right sub-tree
        if type(node[5]) != type([]):
            if prior:
                node[5] = CNET.multiply_into_clt2(node[5], p, assign_idx_new, assign_new1, retain, prior)
            else:
                node[5] = CNET.multiply_into_clt4(p, q_old, assign_idx_new, assign_new1, retain, node[5])
        else:
            self.project_and_update_helper(q_tree[5], p, q_old, assign_idx_new, assign_new1, retain, prior)
    def project_and_update_helper_smooth(self, q_tree, p, q_old, assign_idx, assign, retain, prior):
        # Initialization
        node = q_tree
        id = node[1]
        marg_idx = np.array([False]*2*self.nvariables)
        evid_idx = np.hstack(([False]*self.nvariables,assign_idx))
        # Set parameters
        marg_idx[self.nvariables+id] = True
        marg0, marg1 = np.array([0]), np.array([1])
        a = assign_idx[:id].sum()
        assign_idx_new = np.hstack((assign_idx[:id],np.array([True]),assign_idx[id+1:]))
        assign_new0 = np.hstack((assign[:a],marg0,assign[a:]))
        assign_new1 = np.hstack((assign[:a],marg1,assign[a:]))
        evid_idx[id] = True
        evid0 = np.hstack((marg0, assign))
        evid1 = np.hstack((marg1, assign))
        # Adjust parameters for prior update
        if prior:
            marg_idx = marg_idx[self.nvariables:]
            evid_idx = np.array([False]*self.nvariables)
            evid0, evid1 = np.array([]), np.array([])
        # Perform inference on current and transition distributions
        q0 = q_old.prob_evid(q_old.tree,marg_idx[self.nvariables:],marg0)
        q1 = 1-q0
        p0_0 = p.prob_marg_evid(evid_idx,evid0,marg_idx,marg0)
        p0_1 = p.prob_marg_evid(evid_idx,evid1,marg_idx,marg0)
        p1_0 = 1-p0_0
        p1_1 = 1-p0_1
        # Update branch parameters for Q(Xt_pid=0|~Xt_asgn)
        node[2] = (q0*p0_0) + (q1*p0_1)
        # If left branch is CLT update, otherwise update left sub-tree
        if type(node[4]) != type([]):
            if prior:
                node[4] = CNET.multiply_into_clt2(node[4], p, assign_idx_new, assign_new0, retain, prior)
            else:
                node[4] = CNET.multiply_into_clt4_smooth(p, q_old, assign_idx_new, assign_new0, retain, node[4])
        else:
            self.project_and_update_helper_smooth(q_tree[4], p, q_old, assign_idx_new, assign_new0, retain, prior)
        # Update branch parameters for Q(Xt_pid=1|~Xt_asgn)
        node[3] = 1 - node[2]
        # If right branch is CLT update, otherwise update right sub-tree
        if type(node[5]) != type([]):
            if prior:
                node[5] = CNET.multiply_into_clt2(node[5], p, assign_idx_new, assign_new1, retain, prior)
            else:
                node[5] = CNET.multiply_into_clt4_smooth(p, q_old, assign_idx_new, assign_new1, retain, node[5])
        else:
            self.project_and_update_helper_smooth(q_tree[5], p, q_old, assign_idx_new, assign_new1, retain, prior)
    def project_and_update(self, p, retain=False, prior=False):
        q_old = copy.deepcopy(self)
        assign_idx = np.array([False]*self.nvariables)
        assign = np.array([],dtype=int)
        self.project_and_update_helper(self.tree, p, q_old, assign_idx, assign, retain, prior)
    def project_and_update_smooth(self, p, retain=False, prior=False):
        q_old = copy.deepcopy(self)
        assign_idx = np.array([False]*self.nvariables)
        assign = np.array([],dtype=int)
        self.project_and_update_helper_smooth(self.tree, p, q_old, assign_idx, assign, retain, prior)
    '''
        Slightly perturb the structure of a cutset network by slightly
        perturbing each leaf CLT by swapping out K edges with a perturbation
        probability of perturb_prob and return a new cutset network
    '''
    def perturb_helper(self, tree, K=1, perturb_prob=0.5):
        node = tree
        # Process the left branch
        if type(node[4]) != type([]):
            if np.random.uniform() <= perturb_prob:
                node[4] = node[4].perturb(K, 0)
        else:
            self.perturb_helper(node[4], K, perturb_prob)
        # Process the right branch
        if type(node[5]) != type([]):
            if np.random.uniform() <= perturb_prob:
                node[5] = node[5].perturb(K, 0)
        else:
            self.perturb_helper(node[5], K, perturb_prob)
    def perturb(self, K=1, perturb_prob=0.5, seed=0):
        # Create copy of current cutset network to perturb
        cnet = copy.deepcopy(self)
        # Invoke helper
        if seed!=0: np.random.seed(seed)
        if type(cnet.tree) != type([]):
            if np.random.uniform() <= perturb_prob:
                return cnet.tree.perturb(K, 0)
        else:
            cnet.perturb_helper(cnet.tree, K, perturb_prob)
        # Return perturbed cutset network
        return cnet

class DetBernouli:
    def __init__(self,y):
        self.p=0.99
        self.Y=y
    def predict_log_proba(self,dataset):
        if self.Y==0:
            return np.tile(np.array([np.log(self.p),np.log(1.0-self.p)]),(dataset.shape[0],1))
        else:
            return np.tile(np.array([np.log(1.0-self.p),np.log(self.p)]),(dataset.shape[0],1))

    def predict_proba(self, dataset):
        if self.Y==0:
            return np.tile(np.array([self.p,1.0-self.p]),(dataset.shape[0],1))
        else:
            return np.tile(np.array([1.0-self.p,self.p]),(dataset.shape[0],1))



def main_cutset():
    train_filename = sys.argv[1]
    test_filename = train_filename[:-11] + '.test.data'
    valid_filename = train_filename[:-11] + '.valid.data'

    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    #train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    print("Learning Chow-Liu Trees on original data ......")
    clt = CLT()
    clt.learnStructure(train_dataset)
    print("done")

    cnets = []
    print("Learning Cutset Networks.....")
    for i in range(3, 10):
        cnets.append(CNET(depth=i))
    for cnet in cnets:
        cnet.learnStructure(train_dataset)
    print("done")

    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    print('Test set LL scores')
    print(clt.computeLL(test_dataset) / test_dataset.shape[0], "Chow-Liu")
    for cnet in cnets:
        print(cnet.computeLL(test_dataset) / test_dataset.shape[0], cnet.depth)
    print()

    print('Valid set LL scores')
    print(clt.computeLL(valid_dataset) / valid_dataset.shape[0], "Chow-Liu")
    for cnet in cnets:
        print(cnet.computeLL(valid_dataset) / valid_dataset.shape[0], cnet.depth)
    print()




if __name__=="__main__":
    #main_samples()
    main_cutset()
