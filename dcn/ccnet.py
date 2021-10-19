# Learn conditional cutset network

from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import SGDClassifier
#from sklearn.calibration import CalibratedClassifierCV
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import datasets
#import scipy.sparse as sps
from sklearn.neural_network import MLPClassifier
import math
import sys
import copy
import heapq
import itertools
import glob
import re
import CNET
import cPickle as pickle
from dcn.Util2 import *

import dcn.utilM

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

from collections import deque
import time

class count_func:
    def __init__(self, in_arr):
        self.input = in_arr
        #self.out_prob = np.zeros((self.input.shape[0],2))

    def predict_log_proba(self, dummy):
        xcounts = Util2.compute_xcounts(self.input) + 1 # laplace correction
        xprob = Util2.normalize1d(xcounts)
        return np.log(xprob)


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
    def __init__(self):
        self.nvariables = 0
        self.xycounts = np.ones((1, 1, 2, 2), dtype=int)
        self.xcounts = np.ones((1, 2), dtype=int)
        self.xyprob = np.zeros((1, 1, 2, 2))
        self.xprob = np.zeros((1, 2))
        self.topo_order = []
        self.parents = []
        self.Tree = None #shasha, save tree
        self.log_cond_cpt = []  # shasha
        self.save_info = None # Shasha 0709
        self.tree_path = [] # Shasha0806
    '''
        Learn the structure of the Chow-Liu Tree using the given dataset
    '''
    def learnStructure(self, dataset):
        self.nvariables = dataset.shape[1]
        self.xycounts = Util2.compute_xycounts(dataset) + 1 # laplace correction
        self.xcounts = Util2.compute_xcounts(dataset) + 2 # laplace correction
        self.xyprob = Util2.normalize2d(self.xycounts)
        self.xprob = Util2.normalize1d(self.xcounts)
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util2.compute_edge_weights(self.xycounts, self.xcounts) * (-1.0)
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
        #self.Tree = Tree   # Shasha
        #self.get_log_cond_cpt()
    '''
        Learn the structure of the Chow-Liu Tree using the given p_xy and p_x
        Shasha
    '''
    def learnStructure_prob(self, p_xy, p_x):
        self.nvariables = p_x.shape[0]
        self.xyprob = p_xy
        self.xprob = p_x
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util2.compute_MI_prob(self.xyprob, self.xprob) * (-1.0)
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
        #self.Tree = Tree   # Shasha
        #self.get_log_cond_cpt()
    '''
        Learn the structure of the Chow-Liu Tree using the given mutual information
        Shasha
        Used only in specail cases
    '''
    def learnStructure_MI(self, mi):
        self.nvariables = mi.shape[0]
        #self.xyprob = p_xy
        #self.xprob = p_x
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = mi * (-1.0)
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
        #self.Tree = Tree   # Shasha
        #self.get_log_cond_cpt()
    '''
        Update the Chow-Liu Tree using weighted samples
    '''
    def update(self, dataset_, weights=np.array([])):
        # Perform Sampling importance resampling based on weights
        # assume that dataset_.shape[0] equals weights.shape[0] because each example has a weight
        if weights.shape[0]==dataset_.shape[0]:
            norm_weights = Util2.normalize(weights)
            indices = np.argwhere(np.random.multinomial(dataset_.shape[0], norm_weights)).ravel()
            dataset = dataset_[indices, :]
        else:
            dataset=dataset_
            print ("Not using weight to update")
        self.xycounts += Util2.compute_xycounts(dataset)
        self.xcounts += Util2.compute_xcounts(dataset)
        self.xyprob = Util2.normalize2d(self.xycounts)
        self.xprob = Util2.normalize1d(self.xcounts)
        edgemat = Util2.compute_edge_weights(self.xycounts, self.xcounts) * (-1.0)
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
    '''
        Update the Chow-Liu Tree using weighted samples, exact update
    '''
    def update_exact(self, dataset_, weights=np.array([]), structure_update_flag = False):
        # Perform based on weights
        # assume that dataset_.shape[0] equals weights.shape[0] because each example has a weight
        # try to avoid sum(weights = 0
        if weights.shape[0]==dataset_.shape[0] and np.sum(weights > 0):
        #if weights.shape[0]==dataset_.shape[0]:
            #self.xycounts += Util2.compute_weighted_xycounts(dataset_, weights)
            #self.xcounts += Util2.compute_weighted_xcounts(dataset_, weights)
            smooth = max (np.sum(weights), 1.0) / dataset_.shape[0]
            #print ("smooth: ", smooth)
            self.xycounts = Util2.compute_weighted_xycounts(dataset_, weights) + smooth
            self.xcounts = Util2.compute_weighted_xcounts(dataset_, weights) + 2.0 *smooth
        else:
            dataset=dataset_
            print ("Not using weight to update")
            self.xycounts += Util2.compute_xycounts(dataset)
            self.xcounts += Util2.compute_xcounts(dataset)

        self.xyprob = Util2.normalize2d(self.xycounts)
        self.xprob = Util2.normalize1d(self.xcounts)

        if structure_update_flag == True:
            #print ("update structure")
            #self.xyprob = Util2.normalize2d(self.xycounts)
            #self.xprob = Util2.normalize1d(self.xcounts)
            edgemat = Util2.compute_edge_weights(self.xycounts, self.xcounts) * (-1.0)
            Tree = minimum_spanning_tree(csr_matrix(edgemat))
            self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)
    def computeLL(self,dataset):
        prob=0.0

        # shasha0809
        if self.xyprob.shape[0] != dataset.shape[1]:
            return utilM.get_tree_dataset_ll(dataset,self.topo_order, self.parents, self.log_cond_cpt)

        """
        if self.xyprob.shape[0] != dataset.shape[1]:
            #print ('compute using log conditional cpt')
            for i in range(dataset.shape[0]):
                for j in xrange(self.nvariables):
                    x = self.topo_order[j]
                    assignx=dataset[i,x]
                    # if root sample from marginal
                    if self.parents[x] == -9999:
                        prob += self.log_cond_cpt[0, assignx, 0]
                    else:
                        # sample from p(x|y)
                        y = self.parents[x]
                        assigny = dataset[i,y]
                        prob += self.log_cond_cpt[j, assignx, assigny]
            return prob
        """

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
                    prob+=np.log(self.xyprob[x, y, assignx, assigny] / self.xprob[y, assigny])
        return prob
    '''
        Chiro: Given evidence, find the MAP (assignment,value) of the CLT
    '''
    def map(self,evid_arr):
        nvars = self.topo_order.shape[0]
        map_assign, map_val = np.zeros(nvars,dtype=int), 0.00
        evid_expand = np.zeros((1,evid_arr.shape[1]+1), dtype=int)
        evid_expand[:,:-1] = evid_arr[0]
        for v in xrange(nvars):
            child = self.topo_order[v]
            parent = self.parents[child]
            # Chiro
            # print("Parent, Child, Func:",parent,child,self.func_list[child])
            if parent == -9999:
                X = evid_arr
            else:
                X = evid_expand
                X[-1] = map_assign[parent]
            Y = self.func_list[child].predict_log_proba(X)
            map_assign[child], map_val = Y.argmax(), map_val+Y.max()
            # Chiro
            # print("Map Assign, Map Val:",map_assign[child],map_val)
        return math.exp(map_val), map_assign

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
                    prob=self.xyprob[x, y, 0, assigny] / self.xprob[y, assigny]
                    samples[i, x] = int(np.random.random() > prob)
        return samples
    """
    # Shasha's code
    """
    def get_log_cond_cpt(self):
        # pairwised egde CPT in log format based on tree structure

        self.log_cond_cpt = np.log(Util2.compute_conditional_CPT(self.xyprob,self.xprob,self.topo_order, self.parents))
        #print (self.log_cond_cpt)
    """
    def get_cond_cpt(self):

        self.cond_cpt = np.exp(self.log_cond_cpt)
    """
    def getWeights(self, samples):

        self.get_log_cond_cpt()

        probs = utilM.get_sample_ll(samples,self.topo_order, self.parents, self.log_cond_cpt)
        return probs
    # find the path from each node to root
    def get_tree_path(self):

        self.tree_path.append([0])
        for i in xrange(1,self.nvariables):
            #print i
            single_path = []
            single_path.append(i)
            curr = i
            while curr!=0:
                curr = self.parents[curr]
                single_path.append(curr)

            #print single_path
            self.tree_path.append(single_path)
    # store the pairwised edge potentials. Same sequence as self.parents
    #def get_edge_potential(self):
    # set the evidence
    def instantiation(self, evid_list):
        #print ('in instantiation')
        #print (evid_list)


        cond_cpt = np.exp(self.log_cond_cpt)
        #print ("before:")
        #print (cond_cpt)
        for i in xrange (len(evid_list)):
            variable_id = evid_list[i][0]
            value = evid_list[i][1]

            index_c = np.where(self.topo_order==variable_id)[0][0]
            # variable as parent
            varible_child = np.where(self.parents ==variable_id)[0]
            ix = np.isin(self.topo_order, varible_child)
            index_p = np.where(ix)[0]
            #print (index_p)

            # set varible value = 0
            if value == 0:
                cond_cpt[index_c, 1,:] = 0
                cond_cpt[index_p, :,1] = 0

            # set varible value = 1
            elif value == 1:
                cond_cpt[index_c, 0,:] = 0
                cond_cpt[index_p, :,0] = 0

            else:
                print ('error in value: ', value)
                exit()
            #print ("after: ")
            #print (cond_cpt)

        return cond_cpt
    def inference(self, cond_cpt, ids):

        #return utilM.get_prob_matrix(self.topo_order, self.parents, cond_cpt, ids)
        return utilM.get_prob_matrix(self.topo_order, self.parents, cond_cpt, ids, self.tree_path)
    def getWeightFun(self,evid_arr, query_arr, function_type, alpha):
        self.func_list = []  # based on topo order
        #print ('topo order: ', self.topo_order)
        #print ('parent: ', self.parents)
        #print ('evid:')
        #print (evid_arr)
        #print ('query:')
        #print (query_arr)
        h_size = max(10, evid_arr.shape[1]+1)
        for i in xrange(self.topo_order.shape[0]):
            child = self.topo_order[i]
            parent = self.parents[child]
            # the root node
            #print ('child: ',child, 'parent: ', parent)
            if parent ==  -9999:
                X = evid_arr
            else:
                #X = np.append(evid_arr, np.transpose(query_arr[:,parent]), axis =1)
                X = np.zeros((evid_arr.shape[0], evid_arr.shape[1]+1), dtype = int)
                X[:,:-1] = evid_arr
                X[:,-1] = np.transpose(query_arr[:,parent])
            Y = query_arr[:,child]
            #print('X: ')
            #print (X)
            #print ('Y:')
            #print (Y)
            #print ('shape:',Y.shape)

            if function_type == 'LR':
                sum_Y = np.sum(Y)
                # got pure value of Y
                if sum_Y == 0 or sum_Y == Y.shape[0]:

                    func = count_func(Y.reshape(Y.shape[0],1))
                else:
                    func = LogisticRegression(C=1.0, penalty='l1').fit(X, Y)
            elif function_type == 'NN':
                func =  MLPClassifier(activation='logistic',solver='adam', alpha=alpha,
                              hidden_layer_sizes = (h_size,),
                              random_state = 1).fit(X, Y)
            else:
                print ('invalid function')
                exit()

            self.func_list.append(func)
    def computeLLFunc(self,evid_arr, query_arr ):
        prob=0.0
        # add 1 column to the end
        evid_expand = np.zeros((evid_arr.shape[0], evid_arr.shape[1]+1), dtype = int)
        evid_expand[:,:-1] = evid_arr
        for i in range(query_arr.shape[0]):
            for j in xrange(self.topo_order.shape[0]):
                child = self.topo_order[j]
                parent = self.parents[child]
                assign_c=query_arr[i,child]
                # the root node
                if parent ==  -9999:
                    X = evid_arr[i:i+1]
                else:
                    X = evid_expand[i:i+1]
                    X[0,-1] = query_arr[i,parent]
                #print ('X:', X)
                prob += self.func_list[j].predict_log_proba(X)[0, assign_c]
        return prob
    def calc_cond_probs(self, evid_arr):
        nvars = self.topo_order.size
        ind_idx = self.parents == -9999
        ind_vars = self.topo_order[ind_idx]
        zeros = np.zeros(ind_idx.size, dtype=int)
        ones = np.ones(ind_idx.size, dtype=int)
        zeros[ind_idx], ones[ind_idx] = (-1, -1)
        params = np.vstack((np.arange(nvars), zeros, ones)).T
        evids = np.tile(evid_arr, (nvars, 1))
        master0 = np.hstack((params[:, 0].reshape(nvars, 1), evids, params[:, 1].reshape(nvars, 1)))
        master1 = np.hstack((params[:, 0].reshape(nvars, 1), evids, params[:, 2].reshape(nvars, 1)))
        def cond_probs(args):
            i, evidp, evid = args[0], args[1:].reshape(1, args[1:].size), args[1:-1].reshape(1, args[2:].size)
            if evidp[0, -1] != -1:
                return self.func_list[i].predict_log_proba(evidp)
            return self.func_list[i].predict_log_proba(evid)[0]
        pzeros = np.apply_along_axis(cond_probs, 1, master0)
        pones = np.apply_along_axis(cond_probs, 1, master0)
        return np.exp(np.hstack((pzeros, pones)))
    def prob_evid(self, evid_arr, query_idx, query, topo = False):
        if query.size == 0:
            return 1.0
        nvars = self.parents.size
        if self.parents.sum() == -9999 * self.parents.size:
            return self.xprob[np.where(evid_idx == True)[0], evid].prod()
        if self.parents.size != self.topo_order.size:
            self.topo_order = np.append(self.topo_order, np.where(self.parents == -9999)[0][1:])
        if not topo:
            query_temp = np.full(query_idx.size, -1)
            query_temp[query_idx] = query
            query_idx = np.array([ query_idx[v] for v in self.topo_order ])
            query = np.array([ query_temp[v] for v in self.topo_order ])[query_idx]
        transdict = dict(zip(self.topo_order, np.arange(nvars)))
        parents = np.array([ (self.parents[v] if self.parents[v] != -9999 else v) for v in self.topo_order ])
        parents_order = np.array([ transdict.get(p) for p in parents ])
        cond_probs = self.calc_cond_probs(evid_arr)
        all_query = np.vstack((1 - query, query, 1 - query, query)).T.astype(float)
        cond_probs[query_idx] = all_query * cond_probs[query_idx]
        query_dict = dict(zip(self.topo_order[query_idx], query))
        parents_query = np.array([ query_dict.get(p, -9999) for p in parents ])
        query_both_idx = query_idx * (parents_query != -9999)
        query_selector = np.vstack((1 - parents_query[query_both_idx], 1 - parents_query[query_both_idx], parents_query[query_both_idx], parents_query[query_both_idx])).T
        cond_probs[query_both_idx] = cond_probs[query_both_idx] * query_selector
        cond_probs[query_both_idx] = np.tile(np.vstack((cond_probs[query_both_idx, ::2].sum(axis=1), cond_probs[query_both_idx, 1::2].sum(axis=1))).T, 2)
        for v in reversed(xrange(1, cond_probs.shape[0])):
            sum_p0, sum_p1 = cond_probs[v, 0:2].sum(), cond_probs[v, 2:4].sum()
            cond_probs[parents_order[v], ::2] *= sum_p0
            cond_probs[parents_order[v], 1::2] *= sum_p1
        return max(cond_probs[0, 0:2].sum(), cond_probs[0, 2:4].sum())
    def convert_to_clt(self, evid_arr):
        clt = CNET.CLT(self)
        nvars = clt.topo_order.size
        clt.xycounts = np.ones(clt.xycounts.shape)
        clt.xcounts = np.ones(clt.xcounts.shape)
        evid_arrx = evid_arr.reshape(1, evid_arr.size)
        evid_arr0 = np.hstack((evid_arrx, np.array([[0]])))
        evid_arr1 = np.hstack((evid_arrx, np.array([[1]])))
        for v in xrange(self.topo_order.size):
            x = self.topo_order[v]
            y = self.parents[x] if self.parents[x] != -9999 else x
            func = self.func_list[v]
            if x == y:
                clt.xycounts[x, y, 0, 0] = np.exp(func.predict_log_proba(evid_arrx)[0, 0])
                clt.xycounts[x, y, 1, 1] = 1.0 - clt.xycounts[x, y, 0, 0]
                clt.xycounts[x, y, 0, 1] = 0.0
                clt.xycounts[x, y, 1, 0] = 0.0
            else:
                clt.xycounts[y, x, 0, :] = np.exp(func.predict_log_proba(evid_arr0))
                clt.xycounts[y, x, 1, :] = np.exp(func.predict_log_proba(evid_arr1))
            clt.xycounts[x, y] = clt.xycounts[y, x].T
            clt.xcounts[x] = clt.xycounts[y, x].sum(axis=0)
        clt.xyprob = Util2.normalize2d(clt.xycounts)
        clt.xprob = Util2.normalize1d(clt.xcounts)
        return clt

class MIXTURE_CLT():

    def __init__(self):
        self.n_components = 0
        self.mixture_weight = None
        # weigths associated with each record in mixture
        # n_componets * n_var
        #self.clt_weights_list = None
        self.clt_list =[]   # chow-liu tree list
    '''
        Learn the structure of the Chow-Liu Tree using the given dataset
    '''
    def learnStructure(self, dataset, n_components):
        print ("Mixture of Chow-Liu Tree ......" )
        # Shuffle the dataset

        self.n_components = n_components
        self.mixture_weight = np.full(n_components , 1.0 /n_components )
        #print ("mixture weights: ", self.mixture_weight)
        data_shuffle = np.copy(dataset)
        np.random.shuffle(data_shuffle)
        n_data = data_shuffle.shape[0] / self.n_components


        for c in xrange(self.n_components):
            if c == self.n_components - 1:   # the last portion
                data_slice = data_shuffle[c*n_data : , : ]

            else:
                data_slice = data_shuffle[c*n_data: ((c+1)*n_data), :]

            clt = CLT()
            clt.learnStructure(data_slice)

            self.clt_list.append(clt)

    # Learning parameters using EM
    def EM(self, dataset, max_iter, epsilon):

        #print ("epsilon: ", epsilon)
        structure_update_flag = False

        clt_weights_list = np.zeros((self.n_components, dataset.shape[0]))

        ll_score = -np.inf
        ll_score_prev = -np.inf
        for itr in xrange(max_iter):

            #print ( "iteration: ", itr)

            if itr > 0:
                #print (np.einsum('ij->i', clt_weights_list))
                self.mixture_weight = Util2.normalize(np.einsum('ij->i', clt_weights_list) + 1.0)  # smoothing and Normalize
                #mixture_weights = np.sum(clt_weights_list, axis = 1)
                #print (self.mixture_weight)

                # update tree structure: the first 50 iterations, afterward, every 50 iterations
                if itr < 50 or itr % 50 == 0:
                    structure_update_flag = True

                for c in xrange(self.n_components):
                    self.clt_list[c].update_exact(dataset, clt_weights_list[c], structure_update_flag)
                    #self.clt_list[c].update_exact(dataset, clt_weights_list[c])

                structure_update_flag = False

            ll_score_prev = ll_score
            ## E step
            #for c in xrange(self.n_components):
            #    self.clt_weights_list.append(self.clt_list[c].getWeights(dataset) * self.mixture_weight[c])
            #self.clt_weights_list = np.asarray(self.clt_weights_list)

            log_mixture_weights = np.log(self.mixture_weight)
            for c in xrange(self.n_components):
                clt_weights_list[c] = self.clt_list[c].getWeights(dataset) + log_mixture_weights[c]

            #ll_score = np.einsum('ij->', self.clt_weights_list)   # Wrong



            #print ("shape: ", self.clt_weights_list.shape[1])
            # Normalize weights
            # input is in log format, output is in normal
            clt_weights_list, ll_score = Util2.m_step_trick(clt_weights_list)
            #print (self.clt_weights_list)

            #print ("LL score diff : ", ll_score - ll_score_prev)
            if abs(ll_score - ll_score_prev) < epsilon:
                print ("converged")
                break


        print ("Total iterations: ", itr)
        print('Train set LL scores: ', ll_score / dataset.shape[0])
        print ("difference in LL score: ", ll_score - ll_score_prev)


    def computeLL(self, dataset):

        clt_weights_list = np.zeros((self.n_components, dataset.shape[0]))

        log_mixture_weights = np.log(self.mixture_weight)

        for c in xrange(self.n_components):
            clt_weights_list[c] = self.clt_list[c].getWeights(dataset) + log_mixture_weights[c]


        clt_weights_list, ll_score = Util2.m_step_trick(clt_weights_list)

        return ll_score

    def computeLL_each_datapoint(self, dataset):

        clt_weights_list = np.zeros((self.n_components, dataset.shape[0]))

        log_mixture_weights = np.log(self.mixture_weight)

        for c in xrange(self.n_components):
            clt_weights_list[c] = self.clt_list[c].getWeights(dataset) + log_mixture_weights[c]


        ll_scores = Util2.get_ll_trick(clt_weights_list)

        return ll_scores




    def inference(self,evid_list, ids):
        dim = ids.shape[0]
        p_xy_all = np.zeros((dim, dim, 2, 2))
        p_x_all = np.zeros((dim, 2))
        for i, t in enumerate(self.clt_list):
            #print (t.topo_order)
            #print (cond_cpt_list[i])
            #p_xy =  utilM.get_prob_matrix(t.topo_order, t.parents, cond_cpt_list[i], ids)
            if len(evid_list) == 0:  # no evidence
                cond_cpt = np.exp(t.log_cond_cpt)
            else:
                cond_cpt = t.instantiation(evid_list)
            p_xy =  t.inference(cond_cpt, ids)
            p_xy_all += p_xy * self.mixture_weight[i]

        # Normalize
        #p_xy_all = Util2.normalize2d(p_xy_all)
        # Smoothing and renormalize
        #p_xy_all += 1e-8
        #p_xy_all = Util2.normalize2d(p_xy_all)

        p_x_all[:,0] = p_xy_all[0,:,0,0] + p_xy_all[0,:,1,0]
        p_x_all[:,1] = p_xy_all[0,:,0,1] + p_xy_all[0,:,1,1]

        p_x_all[0,0] = p_xy_all[1,0,0,0] + p_xy_all[1,0,1,0]
        p_x_all[0,1] = p_xy_all[1,0,0,1] + p_xy_all[1,0,1,1]


        # Normalize
        #p_x_all = Util2.normalize1d(p_x_all)

        for i in xrange (ids.shape[0]):
            p_xy_all[i,i,0,0] = p_x_all[i,0] - 1e-8
            p_xy_all[i,i,1,1] = p_x_all[i,1] - 1e-8
            p_xy_all[i,i,0,1] = 1e-8
            p_xy_all[i,i,1,0] = 1e-8

        #p_xy_all = Util2.normalize2d(p_xy_all)


        return p_xy_all, p_x_all

# Code copied from Shasha.
# Inefficient and will need to speed up one day
# The mutual information is got from inference of TUM (mixture of CLT)
class CNET_CLT:
    def __init__(self,tree, depth=100):
        self.nvariables=0
        self.depth=depth
        self.tree=tree
        # Chiro
        self.map_assign=None
        self.map_val=None
    def save(self,dir,filename="ccnet"):
        outfilename = dir + '/' + filename + ".model"
        with open(outfilename,'wb') as outfile:
            pickle.dump(self,outfile,pickle.HIGHEST_PROTOCOL)
    def load(self,dir,filename="ccnet"):
        infilename = dir + '/' + filename + ".model"
        with open(infilename,'rb') as infile:
            obj = pickle.load(infile)
            self.nvariables = obj.nvariables
            self.depth = obj.depth
            self.tree = obj.tree
            self.map_assign = obj.map_assign
            self.map_val = obj.map_val
    # The structure is built on query variables
    def learnStructureHelper(self, evid_arr, query_arr, query_ids, function_type, alpha):
        # 'path' is a list, contains the cutset nodes from root to parents, and the corresponding value
        curr_depth=self.nvariables - query_arr.shape[1]
        #print ("curr_depth: ", curr_depth)
        #print ('a', evid_list)
        #n_evid = evid_arr.shape[0]  # number of evidence
        #if True:
        #print ('evid:')
        #print (evid_arr)
        #print ('query:')
        #print (query_arr)

        if query_arr.shape[0]<10 or query_arr.shape[1]<5 or curr_depth >= self.depth:
            clt=CLT()
            clt.learnStructure(query_arr)
            clt.getWeightFun(evid_arr, query_arr, function_type, alpha)

            #print ("built from count")
            #print ("topo_order: ", clt.topo_order)
            #print ("parents: ", clt.parents)
            #print ("pxy: " )
            #for sszezeaei in xrange (clt.xyprob.shape[0]):
            #    print ("-------X = ", i)
            #    for j in xrange (clt.xyprob.shape[1]):
            #        print ("-------Y = ", j)
            #        print (clt.xyprob[i,j,:,:])

            return clt
        xycounts = Util2.compute_xycounts(query_arr) + 1  # laplace correction
        xcounts = Util2.compute_xcounts(query_arr) + 2  # laplace correction
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util2.compute_edge_weights(xycounts, xcounts)
        # reset self mutual information to be 0
        np.fill_diagonal(edgemat, 0) #shasha#

        #print ("edgemat: ", edgemat)
        scores = np.sum(edgemat, axis=0)
        #print (scores)
        variable = np.argmax(scores)
        variable_id = query_ids[variable]
        #print ("variable: ", variable_id)

        # if the selected variable has pure value, do not extend, just retrun clt
        sum_variable = np.sum(query_arr[:,variable])
        if sum_variable == 0 or sum_variable == query_arr.shape[0]:
            clt=CLT()
            clt.learnStructure(query_arr)
            clt.getWeightFun(evid_arr, query_arr, function_type, alpha)
            return clt


        Y = query_arr[:, variable]
        #print ('Y: ', Y)
        #print ('function_type: ', function_type)
        if function_type == 'LR':
            func = LogisticRegression(C=1.0, penalty='l1').fit(evid_arr, Y)
        elif function_type == 'NN':
            func =  MLPClassifier(activation='logistic',solver='adam', alpha=alpha,
                              hidden_layer_sizes = (max(10,evid_arr.shape[1]),),
                              random_state = 1, max_iter=1000).fit(evid_arr, Y)
        else:
            print ('invalid function')
            exit()

        # The left branch
        index_0 = query_arr[:,variable]==0
        new_query_arr0 = np.delete(query_arr[index_0], variable, 1)
        new_evid_arr0 = evid_arr[index_0]
        #new_evid_arr0 = np.append(new_evid_arr0, np.transpose(np.zeros(new_evid_arr0.shape[0], dtype = int)), 1)
        #new_query_arr0 = np.delete(query_arr[query_arr[:, variable] == 0], variable, 1)
        #p0 = float(new_dataset0.shape[0]) +1.0
        #print ('--------LEFT--------')
        #print ('evid:')
        #print (new_evid_arr0)
        #print ('query:')
        #print (new_query_arr0)

        # The right branch
        #index_1 = query_arr[:,variable]==1
        index_1 = ~index_0    # the invert bool array
        new_query_arr1 = np.delete(query_arr[index_1], variable, 1)
        new_evid_arr1 = evid_arr[index_1]
        #new_evid_arr1 = np.append(new_evid_arr1, np.transpose(np.ones(new_evid_arr1.shape[0], dtype = int)), 1)
        #new_query_arr1 = np.delete(query_arr[query_arr[:,variable]==1],variable,1)
        #p1=float(new_dataset1.shape[0])+1.0
        #print ('--------RIGHT--------')
        #print ('evid:')
        #print (new_evid_arr1)
        #print ('query:')
        #print (new_query_arr1)



        new_query_ids = np.delete(query_ids,variable)
        #new_evid_var = np.append(evid_var, variable_id)



        #print ("p0, p1: ", p0, p1)

        #next_depth = curr_depth + 1
        return [variable,variable_id,func,self.learnStructureHelper(new_evid_arr0, new_query_arr0, new_query_ids, function_type, alpha),
                self.learnStructureHelper(new_evid_arr1, new_query_arr1, new_query_ids, function_type, alpha)]
    def learnStructure(self, evid_arr, query_arr, function_type, alpha):

        self.nvariables = query_arr.shape[1]
        query_ids = np.arange(self.nvariables)
        self.tree=self.learnStructureHelper(evid_arr, query_arr, query_ids, function_type, alpha)
    def computeLL(self,evid_arr, query_arr):
        prob = 0.0
        for i in range(query_arr.shape[0]):
            node=self.tree
            query_ids=np.arange(self.nvariables)
            # Chiro
            # print('[CHIRO] Starting MAP procedure')
            # map_assign, map_val = self.map(evid_arr[i:i+1])
            # raw_input('MAP Finished! Press any key to continue!')
            while isinstance(node,list):
                id,x,func,node0,node1=node
                assignx=query_arr[i,x]
                #print ('prob:', func.predict_log_proba (evid_arr))
                #print ('ass: ', assignx)
                #print ('prob:', func.predict_log_proba (evid_arr[i:i+1]))
                prob += func.predict_log_proba (evid_arr[i:i+1])[0,assignx]
                #print (prob)
                query_ids=np.delete(query_ids,id,0)
                if assignx==1:
                    #prob+=np.log(p1)
                    node=node1
                else:
                    #prob+=np.log(p0)
                    node = node0

            #print ('prob: ', prob)
            # leaf node
            prob+=node.computeLLFunc( evid_arr[i:i+1], query_arr[i:i+1,query_ids])
            # Chiro
            # map_assign, map_val = node.map(evid_arr[i:i+1])
            # print(map_assign,map_val)
            # raw_input('MAP Done!')
        return prob
    '''
        Chiro: Helper function for CCNET k-MAP
    '''
    def kmap_helper(self,evid_arr,node,particles,curr_map_assign,curr_map_val,lvl):
        if not isinstance(node,list):
            # Calculate MAP for leaf node
            leaf_map_val, leaf_map_assign = node.map(evid_arr)
            leaf_map_val = math.log(leaf_map_val)
            # Merge current MAP values with leaf MAP values
            self.map_val, self.map_assign = curr_map_val, np.copy(curr_map_assign)
            Util2.merge(self.map_assign,leaf_map_assign,-1)
            self.map_val += leaf_map_val
            # Push particle into priority queue
            pid = len(particles) + 1
            heapq.heappush(particles,(-self.map_val, pid, self.map_assign))
        else:
            # Unpack node information
            name, id, func, node0, node1 = node
            # Calculate P(Y|X) where X=evidence
            Y = func.predict_log_proba(evid_arr)
            # Set current variable to 0 and traverse the left subtree
            curr_map_assign[id] = 0
            curr_map_val += Y[0,0]
            self.kmap_helper(evid_arr,node0,particles,curr_map_assign,curr_map_val,lvl+1)
            # Set current variable to 1 and traverse the right subtree
            curr_map_assign[id] = 1
            curr_map_val += (Y[0,1]-Y[0,0])
            self.kmap_helper(evid_arr,node1,particles,curr_map_assign,curr_map_val,lvl+1)
            # Reset values of curr_map_assign and curr_map_val
            curr_map_assign[id] = -1
            curr_map_val -= Y[0,1]
    '''
        Chiro: k-MAP function (assignment,value) for CCNET
        All particles will be stored in the form (likelihood, id, assignment)
        The id is to avoid a ValueError whenever the assignments are the same
    '''
    def kmap(self,evid_arr,k=1):
        # Initialize variables
        node = self.tree
        particles = []
        curr_map_val, self.map_val = 0.00, -np.inf
        curr_map_assign, self.map_assign = np.full(self.nvariables,-1,dtype=int), np.full(self.nvariables,-1,dtype=int)
        # Get particles
        self.kmap_helper(evid_arr,node,particles,curr_map_assign,curr_map_val,1)
        # Return the top-k particles
        k_particles = []
        for p in xrange(len(particles)):
            particle = heapq.heappop(particles)
            k_particles.append((math.exp(-particle[0]),particle[1],particle[2]))
        return k_particles
    def map(self,evid_arr):
        map_particle = self.kmap(evid_arr,1)[0]
        return (map_particle[0], map_particle[2])
    def convert_to_cnet_helper(self, tree, evid_arr):
        node = tree
        if type(node) != type([]):
            clt = node.convert_to_clt(evid_arr)
            return clt
        func = node[2]
        prob0 = func.predict_proba(evid_arr.reshape(1, evid_arr.size))[0, 0]
        prob1 = 1 - prob0
        left_tree = self.convert_to_cnet_helper(node[3], evid_arr)
        right_tree = self.convert_to_cnet_helper(node[4], evid_arr)
        return [node[0], node[1], prob0, prob1, left_tree, right_tree]
    def convert_to_cnet(self, evid_arr):
        cnet = CNET.CNET(self.depth)
        cnet.nvariables = self.nvariables
        evid_arr = evid_arr[np.newaxis,:] if evid_arr.ndim == 1 else evid_arr
        cnet.tree = self.convert_to_cnet_helper(self.tree, evid_arr)
        return cnet

class DCCN:
    def __init__(self, max_depth):
        self.sensor_model = None
        self.transition_model = None
        self.nvars = None
        self.max_depth = max_depth
    def fetch_latest_model(self,dir,name,suffix,smooth=False,ccnet=False):
        filename = dir + '/' + name + "." + suffix + ".*.model"
        print('[DEBUG] filename: '+filename)
        all_files = glob.glob(filename)
        if len(all_files)==0: return (-1,0)
        all_i = np.sort(np.array([ int(f.split(".")[-2]) for f in all_files ]))
        i = all_i[-1] if not smooth else all_i[0]
        q = CNET.CNET() if not ccnet else CNET_CLT([], self.max_depth)
        q.load(dir,name+"."+suffix+"."+str(i))
        return (i,q)
    def train(self, dataset_dir, data_name, evid_percent, function_type, alpha, test = False, save = False, restore = True):
        sensor_filename = dataset_dir + data_name + '.sensor'
        transition_filename = dataset_dir + data_name + '.transition'
        sensor_dataset = np.loadtxt(sensor_filename, dtype=int, delimiter=',')
        transition_dataset = np.loadtxt(transition_filename, dtype=int, delimiter=',')
        evid_var_sensor = read_evidence_file(dataset_dir, evid_percent, data_name + '.sensor')
        evid_var_transition = read_evidence_file(dataset_dir, evid_percent, data_name + '.transition')
        query_var_sensor = np.setdiff1d(np.arange(sensor_dataset.shape[1]), evid_var_sensor)
        query_var_transition = np.setdiff1d(np.arange(transition_dataset.shape[1]), evid_var_transition)
        evid_arr_sensor = sensor_dataset[:, evid_var_sensor]
        evid_arr_transition = transition_dataset[:, evid_var_transition]
        query_arr_sensor = sensor_dataset[:, query_var_sensor]
        query_arr_transition = transition_dataset[:, query_var_transition]
        self.nvars = query_var_sensor.shape[0]
        print('*** Training Sensor Model')
        depth = -1
        if restore:
            print('Checking if Sensor Model exists...')
            depth, sensor_model = self.fetch_latest_model(dataset_dir+"/models/",data_name,'sensor',ccnet=True)
            if depth>-1:
                print('Found previous sensor model!')
                self.sensor_model = sensor_model
        if depth == -1:
            print('Training Sensor Model from scratch')
            self.sensor_model = CNET_CLT([], self.max_depth)
            self.sensor_model.learnStructure(evid_arr_sensor, query_arr_sensor, function_type, alpha)
            if save: self.sensor_model.save(dataset_dir+"models/",data_name + '.sensor.' + str(self.max_depth))
        print('*** Training Transition Model')
        depth = -1
        if restore:
            print('Checking if Transition Model exists...')
            depth, transition_model = self.fetch_latest_model(dataset_dir+"/models/",data_name,'transition',ccnet=True)
            if depth>-1:
                print('Found previous transition model!')
                self.transition_model = transition_model
        if depth == -1:
            print('Training Transition Model from scratch')
            self.transition_model = CNET_CLT([], self.max_depth)
            self.transition_model.learnStructure(evid_arr_transition, query_arr_transition, function_type, alpha)
            if save: self.transition_model.save(dataset_dir+"models/",data_name + '.transition.' + str(self.max_depth))
        if test:
            print('*** Testing Models')
            sensor_score = self.sensor_model.computeLL(evid_arr_sensor, query_arr_sensor) / sensor_dataset.shape[0]
            print('Sensor LL score:', sensor_score)
            transition_score = self.transition_model.computeLL(evid_arr_transition, query_arr_transition) / transition_dataset.shape[0]
            print('Transition LL score:', transition_score)
        print('*** Finished Training DCCN')
    def filter_map(self, dataset_dir, results_dir, data_name, evid_percent = '', partition=-1, structure=0, save=False, sample=False, restore=True, sensor_only = False, cltlearn = 999999999):
        print('*** Filtering dataset using Forward Algorithm')
        if save: print('Save models = ON')
        if sample: print('Save Posterior samples = ON')
        if restore: print('Restore from previous model = ON')
        # 1. Load test set
        partition_str = '' if partition==-1 else '.' + str(partition)
        evid_filename = dataset_dir + data_name + partition_str + '.evidence'
        map_filename = results_dir + data_name + partition_str + '.ep.' + str(structure) + '.f.predict'
        evid_dataset = np.loadtxt(evid_filename, dtype=int, delimiter=',')
        nrows, nvars = evid_dataset.shape[0], evid_dataset.shape[1]
        # 2. Learn projection structure
        print('*** Creating Projection Structure Q')
        if structure==0:
            print('*** Using Prior Structure')
            prior = self.sensor_model.convert_to_cnet(evid_dataset[0])
            q = prior.learn_project_structure(self.nvars)
        else:
            print('*** Using Skewed Posterior Structure')
            transition_model = self.transition_model.convert_to_cnet(evid_dataset[0])
            reduced_transition_model = copy.deepcopy(transition_model)
            reduced_transition_model.reduce_dist(self.nvars)
            q = reduced_transition_model.learn_project_structure(self.nvars)
        # 3. Restore from previous model, if restore flag is ON
        start = 1
        if restore:
            print('*** Searching for Previous Models')
            i, m = self.fetch_latest_model(results_dir+"models/",data_name+'.'+str(partition),"f")
            if i>0:
                print('*** Continuing from Time Slice '+str(i))
                start = i
                q = m
        # 4. Filter
        print('*** Performing Filtering')
        with open(map_filename, 'w') as map_file:
            # 4a. Project prior unless restoring from previous distribution
            if start==1:
                print('Projecting prior')
                prior = self.sensor_model.convert_to_cnet(evid_dataset[0])
                q.project_and_update(prior, prior=True)
                print('[DEBUG] evd: '+str(evid_dataset[0]))
                print('[DEBUG] map: '+','.join(map(str, q.map()[1])))
                if save: q.save(results_dir+"models/",data_name+'.'+str(partition)+".f.1")
                if sample: compare_dists_dataset(prior,q,results_dir+"samples/","all.samples.txt",results_dir+"samples/",q_name="q"+partition_str+".1.f")
                map_file.write(','.join(map(str, q.map()[1])) + '\n')
            # 4b. Project time updates
            for e in xrange(start, evid_dataset.shape[0]):
                print('Started processing point', e + 1)
                pt = self.transition_model.convert_to_cnet(evid_dataset[e])
                pt_red = copy.deepcopy(pt)
                pt_red.reduce_dist(self.nvars)
                q_old = copy.deepcopy(q)
                retain = True if cltlearn<=0 else not((e+cltlearn-1)%cltlearn==0)
                if not retain: print('[DEBUG] Structure Learning Step')
                q.project_and_update(pt, retain)
                if save: q.save(results_dir+"models/",data_name+'.'+str(partition)+".f."+str(e+1))
                if sample: compare_dists_dataset(pt,q,results_dir+"samples/","all.samples.txt",results_dir+"samples/",q_name="q"+partition_str+"."+str(e+1)+".f")
                print('[DEBUG] evd: '+str(evid_dataset[e]))
                print('[DEBUG] map: '+','.join(map(str, q.map()[1])))
                map_file.write(','.join(map(str, q.map()[1])) + '\n')
        print('*** Finished Filtering!')
    def smooth_map(self, dataset_dir, results_dir, data_name, evid_percent = '', partition=0, structure=0, save=False, sample=False, restore=True, sensor_only = False, cltlearn=999999999):
        print('*** Smoothing dataset using Backward Algorithm')
        if save: print('Save models = ON')
        if sample: print('Save Posterior samples = ON')
        if restore: print('Restore from previous model = ON')
        # 1. Load test set
        partition_str = '' if partition==-1 else '.' + str(partition)
        evid_filename = dataset_dir + data_name + partition_str + '.evidence'
        map_filename = results_dir + data_name + partition_str + '.ep.' + str(structure) + '.predict'
        evid_dataset = np.loadtxt(evid_filename, dtype=int, delimiter=',')
        nrows, nvars = evid_dataset.shape[0], evid_dataset.shape[1]
        # 2. Learn projection structure
        print('*** Creating Projection Structure Q')
        if structure==0:
            print('*** Using Prior Structure')
            prior = self.sensor_model.convert_to_cnet(evid_dataset[0])
            q = prior.learn_project_structure(self.nvars)
        else:
            print('*** Using Skewed Posterior Structure')
            transition_model = self.transition_model.convert_to_cnet(evid_dataset[0])
            reduced_transition_model = copy.deepcopy(transition_model)
            reduced_transition_model.reduce_dist(self.nvars)
            q = reduced_transition_model.learn_project_structure(self.nvars)
        # 3. Restore from previous model, if restore flag is ON
        start = evid_dataset.shape[0]-2
        if restore:
            print('*** Searching for Previous Models')
            i, m = self.fetch_latest_model(results_dir+"models/",data_name+'.'+str(partition),"s")
            if i>0:
                print('*** Continuing from Time Slice '+str(i))
                start = i
                q = m
        # 4. Filter
        print('*** Performing Smoothing')
        with open(map_filename, 'w') as map_file:
            # 4a. Project prior unless restoring from previous distribution
            if start==evid_dataset.shape[0]-2:
                print('Projecting prior')
                prior = self.sensor_model.convert_to_cnet(evid_dataset[-1])
                q.project_and_update_smooth(prior, prior=True)
                print('[DEBUG] evd: '+str(evid_dataset[-1]))
                print('[DEBUG] map: '+','.join(map(str, q.map()[1])))
                if save: q.save(results_dir+"models/",data_name+'.'+str(partition)+".s."+str(start+2))
                if sample: compare_dists_dataset(prior,q,results_dir+"samples/","all.samples.txt",results_dir+"samples/",q_name="q"+partition_str+"."+str(start+2)+".s")
                map_file.write(','.join(map(str, q.map()[1])) + '\n')
            # 4b. Project time updates
            for e in xrange(start, -1,-1):
                print('Started processing point', e + 1)
                pt = self.transition_model.convert_to_cnet(evid_dataset[e+1])
                pt_red = copy.deepcopy(pt)
                pt_red.reduce_dist(self.nvars)
                retain = not((e+cltlearn-1)%cltlearn==0)
                if not retain: print('[DEBUG] Structure Learning Step')
                q.project_and_update(pt, retain)
                if save: q.save(results_dir+"models/",data_name+'.'+str(partition)+".s."+str(e+1))
                if sample: compare_dists_dataset(pt,q,results_dir+"samples/","all.samples.txt",results_dir+"samples/",q_name="q"+partition_str+"."+str(e+1)+".s",smooth=True)
                print('[DEBUG] evd: '+str(evid_dataset[e]))
                print('[DEBUG] map: '+','.join(map(str, q.map()[1])))
                map_file.write(','.join(map(str, q.map()[1])) + '\n')
        print('*** Finished Smoothing!')
    def particle_filter(self, dataset_dir, results_dir, data_name, evid_percent = '', partition=-1, k=1):
        print('*** Finding MAP using Particle Filtering Algorithm')
        # 1. Load test set
        partition_str = '' if partition==-1 else '.' + str(partition)
        evid_filename = dataset_dir + data_name + partition_str + '.evidence'
        map_filename = results_dir + data_name + partition_str + '.pf.predict'
        evid_dataset = np.loadtxt(evid_filename, dtype=int, delimiter=',')
        nrows, nvars = evid_dataset.shape[0], evid_dataset.shape[1]
        # 2. Particle Filter
        print('*** Performing Particle Filtering')
        with open(map_filename, 'w') as map_file:
            prior = self.sensor_model.convert_to_cnet(evid_dataset[0])
            x_prevs = prior.kmap(k)
            print('[DEBUG] evd: '+str(evid_dataset[0]))
            print('[DEBUG] map: '+','.join(map(str, x_prevs[0][2])))
            map_file.write(','.join(map(str, x_prevs[0][2]))+ '\n')
            for e in xrange(1, evid_dataset.shape[0]):
                print('Started processing point', e + 1)
                transition = self.transition_model.convert_to_cnet(evid_dataset[e])
                x_currs = []
                for j in xrange(len(x_prevs)):
                    x_prev = x_prevs[j]
                    evid_idx = np.array(([False]*self.nvars)+([True]*self.nvars))
                    evid = x_prev[2].copy()
                    x_curr = transition.kmap(k,evid_idx,evid,j+1,self.nvars,log_flag=True)
                    x_currs = list(heapq.merge(x_currs,x_curr))
                x_prevs = x_currs[:k]
                print('[DEBUG] evd: '+str(evid_dataset[e]))
                print('[DEBUG] map: '+','.join(map(str, x_prevs[0][2])))
                map_file.write(','.join(map(str, x_prevs[0][2])) + '\n')
            print('*** Finished Particle Filtering!')


#    # computer the log likelihood score for each datapoint in the dataset
#    # returns a numpy array
#    def computeLL_each_datapoint(self,dataset):
#        probs = np.zeros(dataset.shape[0])
#        for i in range(dataset.shape[0]):
#            prob = 0.0
#            node=self.tree
#            ids=np.arange(self.nvariables)
#            while isinstance(node,list):
#                id,x,p0,p1,node0,node1=node
#                assignx=dataset[i,x]
#                ids=np.delete(ids,id,0)
#                if assignx==1:
#                    prob+=np.log(p1)
#                    node=node1
#                else:
#                    prob+=np.log(p0)
#                    node = node0
#            prob+=node.computeLL(dataset[i:i+1,ids])
#            probs[i] = prob
#        return probs
#
#    def update(self,node, ids, dataset, lamda):
#
#        #node=self.tree
#        #ids=np.arange(self.nvariables)
#
#        if isinstance(node,list):
#            id,x,p0,p1,node0,node1=node
#            p0_index=2
#            p1_index=3
#
#
#            new_dataset1=np.delete(dataset[dataset[:,id]==1],id,1)
#            new_dataset0 = np.delete(dataset[dataset[:, id] == 0], id, 1)
#
#            new_p1 = (float(new_dataset1.shape[0]) + 1.0) / (dataset.shape[0] + 2.0) # smoothing
#            new_p0 = 1 - new_p1
#
#            node[p0_index] = (1-lamda) * p0 + lamda * new_p0
#            node[p1_index] = (1-lamda) * p1 + lamda * new_p1
#
#
#
#            ids=np.delete(ids,id,0)
#
#            self.update (node0, ids, new_dataset0, lamda)
#            self.update (node1, ids, new_dataset1, lamda)
#        else:
#            return
#

class HMM_CNET:
    def __init__(self, max_depth, generative=False):
        self.sensor_model = None
        self.transition_model = None
        self.bitransition_model = None
        self.max_depth = max_depth
        self.generative = generative

    def train(self, dataset_dir, data_name, evid_percent, function_type, alpha, bitrans=False, test=False):
        # SENSOR MODEL
        sensor_filename = dataset_dir + '/' + data_name + '.sensor'
        sensor_dataset = np.loadtxt(sensor_filename, dtype=int, delimiter=',')
        if self.generative:
            print('*** Training Sensor Model')
            self.sensor_model = CNET.CNET(self.max_depth)
            self.sensor_model.learnStructure(sensor_dataset)
        else:
            evid_var_sensor = read_evidence_file(dataset_dir, evid_percent, data_name + '.sensor')
            query_var_sensor =  np.setdiff1d(np.arange(sensor_dataset.shape[1]), evid_var_sensor)
            evid_arr_sensor = sensor_dataset[:,evid_var_sensor]
            query_arr_sensor = sensor_dataset[:,query_var_sensor]
            print('*** Training Sensor Model')
            self.sensor_model = CNET_CLT([],self.max_depth)
            self.sensor_model.learnStructure(evid_arr_sensor, query_arr_sensor, function_type, alpha)

        # TRANSITION MODEL
        transition_filename = dataset_dir + '/' + data_name + '.transition'
        transition_dataset = np.loadtxt(transition_filename, dtype=int, delimiter=',')
        evid_var_transition = read_evidence_file(dataset_dir, evid_percent, data_name + '.transition')
        query_var_transition =  np.setdiff1d(np.arange(transition_dataset.shape[1]), evid_var_transition)
        evid_arr_transition = transition_dataset[:,evid_var_transition]
        query_arr_transition = transition_dataset[:,query_var_transition]
        print('*** Training Transition Model')
        self.transition_model = CNET_CLT([],self.max_depth)
        self.transition_model.learnStructure(evid_arr_transition, query_arr_transition, function_type, alpha)

        # BITRANSITION MODEL (OPTIONAL)
        if bitrans:
            bitransition_filename = dataset_dir + '/' + data_name + '.bitransition'
            bitransition_dataset = np.loadtxt(bitransition_filename, dtype=int, delimiter=',')
            evid_var_bitransition = read_evidence_file(dataset_dir, evid_percent, data_name + '.bitransition')
            query_var_bitransition =  np.setdiff1d(np.arange(bitransition_dataset.shape[1]), evid_var_bitransition)
            evid_arr_bitransition = bitransition_dataset[:,evid_var_bitransition]
            query_arr_bitransition = bitransition_dataset[:,query_var_bitransition]
            print('*** Training Bi-Transition Model')
            self.bitransition_model = CNET_CLT([],self.max_depth)
            self.bitransition_model.learnStructure(evid_arr_bitransition, query_arr_bitransition, function_type, alpha)

        # TEST MODELS (OPTIONAL)
        if test:
            print('*** Testing Models')
            if self.generative:
                sensor_score = self.sensor_model.computeLL(sensor_dataset) / sensor_dataset.shape[0]
            else:
                sensor_score = self.sensor_model.computeLL(evid_arr_sensor, query_arr_sensor) / sensor_dataset.shape[0]
            print('Sensor LL score:', sensor_score)
            transition_score = self.transition_model.computeLL(evid_arr_transition, query_arr_transition) / transition_dataset.shape[0]
            print('Transition LL score:', transition_score)
            if bitrans:
                bitransition_score = self.bitransition_model.computeLL(evid_arr_bitransition, query_arr_bitransition) / bitransition_dataset.shape[0]
                print('Bi-Transition LL score:', bitransition_score)

        print('*** Finished Training HMM_CNET')

    def filter_map(self, dataset_dir, results_dir, data_name, evid_percent="", k=1, l=3, m=100, sensor_only=False):
        print('*** Filtering dataset using Forward Algorithm')
        print('k = ',k,', l = ',l,', m = ',m)
        if sensor_only : print('NOTE: Using only sensor model')
        video_delims = read_video_file(dataset_dir, data_name) # where videos start and end in dataset
        evid_filename = dataset_dir + data_name + '.evidence' # where evidence dataset is stored
        map_files = []
        prob_files = []
        comp_files = []
        # Initialize output files to be written to disk
        for f in xrange(l):
            filename = results_dir + data_name + '.' + str(k) + '.' + str(m) + '.' + str(f+1)
            map_files.append(open(filename + '.predict',"w"))
            prob_files.append(open(filename + '.prob',"w"))
            comp_files.append(open(filename + '.comp',"w"))
        # map_filename = results_dir + data_name + '.' + str(k) + '.' + str(m) + '.predict' # where final map dataset is stored
        evid_dataset = np.loadtxt(evid_filename, dtype=int, delimiter=',')
        nrows, nvars = evid_dataset.shape[0], evid_dataset.shape[1]
        # Start processing each video
        for v in xrange(video_delims.shape[0]-1):
            print('** Processing Video',v+1,'of',video_delims.shape[0])
            start, end = video_delims[v], video_delims[v+1]
            evid_arr = evid_dataset[start]
            # BEGIN PARTICLE GENERATION PIPELINE
            # First, generate m samples from the grounded cutset network
            particles = self.sensor_model.convert_to_cnet(evid_arr).generate_samples(m,sort_samples=True)
            # Next, choose the top-k particles to send forward to the next time slice
            k_prev_particles = particles[-k:]
            # In order to generate the explanations, we need to get the indices for the unique k_particles
            particles_only = np.array(np.array(particles,dtype=object)[:,2].tolist())
            _, inverse = np.unique(particles_only,axis=0,return_inverse=True)
            diff_indexes = np.where(np.abs(inverse - np.hstack((0,inverse[:-1])))>0)[0]
            # If no explanation exists, output this value
            empty_particle = (0.0,0,np.zeros(particles[0][2].size,dtype=int))
            # Now, we need to write down each explanation
            for f in xrange(len(map_files)):
                # Choose each unique particle in descending order of likelihood
                # If the explanation exists
                if f < diff_indexes.size:
                    particle = particles[diff_indexes[-(f+1)]]
                # Otherwise, use an empty explanation
                else:
                    particle = empty_particle
                # Write particle value and probability to disk
                map_files[f].write(','.join(map(str,particle[2]))+"\n")
                prob_files[f].write(str(particle[0])+"\n")
                # Calculate component probabilities
                comp_probs = np.zeros((particle[2]>0).sum())
                comp_indexes = np.where(particle[2]>0)[0]
                sensor_cnet = self.sensor_model.convert_to_cnet(evid_arr)
                for c in xrange(comp_indexes.size):
                    evid_idx = np.array([False]*particle[2].size)
                    evid_idx[comp_indexes[c]] = True
                    comp_probs[c] = sensor_cnet.prob_evid(sensor_cnet.tree,evid_idx,np.ones(1,dtype=int))
                comp_files[f].write(','.join(map(str,comp_probs))+"\n")

            for e in xrange(start+1,end):
                if e%50==0:
                    print('Started processing point',e+1)

                if sensor_only:
                    evid_arr = evid_dataset[e]
                    particles = self.sensor_model.convert_to_cnet(evid_arr).generate_samples(m,sort_samples=True)
                    # In order to generate the explanations, we need to get the indices for the unique k_particles
                    particles_only = np.array(np.array(particles,dtype=object)[:,2].tolist())
                    _, inverse = np.unique(particles_only,axis=0,return_inverse=True)
                    diff_indexes = np.where(np.abs(inverse - np.hstack((0,inverse[:-1])))>0)[0]
                    # If no explanation exists, output this value
                    empty_particle = (0.0,0,np.zeros(particles[0][2].size,dtype=int))
                    # Now, we need to write down each explanation
                    for f in xrange(len(map_files)):
                        # Choose each unique particle in descending order of likelihood
                        # If the explanation exists
                        if f < diff_indexes.size:
                            particle = particles[diff_indexes[-(f+1)]]
                        # Otherwise, use an empty explanation
                        else:
                            particle = empty_particle
                        # Write particle value and probability to disk
                        map_files[f].write(','.join(map(str,particle[2]))+"\n")
                        prob_files[f].write(str(particle[0])+"\n")
                        # Calculate component probabilities
                        comp_probs = np.zeros((particle[2]>0).sum())
                        comp_indexes = np.where(particle[2]>0)[0]
                        sensor_cnet = self.sensor_model.convert_to_cnet(evid_arr)
                        for c in xrange(comp_indexes.size):
                            evid_idx = np.array([False]*particle[2].size)
                            evid_idx[comp_indexes[c]] = True
                            comp_probs[c] = sensor_cnet.prob_evid(sensor_cnet.tree,evid_idx,np.ones(1,dtype=int))
                        comp_files[f].write(','.join(map(str,comp_probs))+"\n")
                    continue

                # For each particle, initialize new cutset network
                new_particles = []
                for p in xrange(len(k_prev_particles)):
                    # First, initialize evidence
                    evid_arr = np.zeros(2*nvars,dtype=int)
                    evid_arr[0:nvars] = k_prev_particles[p][2]
                    evid_arr[nvars:(2*nvars)] = evid_dataset[e:e+1][0]
                    # DEBUG
                    # print('evid_arr: '+str(evid_arr))
                    # print('map: '+str(self.transition_model.convert_to_cnet(evid_arr).map()))
                    # Sample m points from the current cutset network
                    cnet_particles = self.transition_model.convert_to_cnet(evid_arr).generate_samples(m,sort_samples=True,seed_id=(m*p))
                    # DEBUG
                    # print('cnet_particles: '+str(cnet_particles))
                    # raw_input('Press any key to continue...')
                    # Add points to list of (k*m) particles
                    new_particles += cnet_particles
                # Sort new_particles
                # print('new_particles: '+str(new_particles))
                # raw_input('Press any key to continue...')
                new_particles = sorted(new_particles)
                # DEBUG
                # print('new_particles: '+str(new_particles))
                # raw_input('Press any key to continue...')
                # From new_particles, choose top-k particles
                # k_prev_particles = new_particles[-k:]
                # Begin explanation pipeline for new_particles
                # In order to generate the explanations, we need to get the indices for the unique k_particles
                particles_only = np.array(np.array(new_particles,dtype=object)[:,2].tolist())
                _, inverse = np.unique(particles_only,axis=0,return_inverse=True)
                diff_indexes = np.where(np.abs(inverse - np.hstack((0,inverse[:-1])))>0)[0]
                # If no explanation exists, output this value
                empty_particle = (0.0,0,np.zeros(particles[0][2].size,dtype=int))
                # First, we need to write down each explanation
                for f in xrange(len(map_files)):
                    # Choose each unique particle in descending order of likelihood
                    # If the explanation exists
                    if f < diff_indexes.size:
                        d = diff_indexes[-(f+1)]
                        particle = new_particles[d]
                        evid_arr[0:nvars] = k_prev_particles[d/m][2]
                    # Otherwise, use an empty explanation
                    else:
                        particle = empty_particle
                    # Write particle value and probability to disk
                    map_files[f].write(','.join(map(str,particle[2]))+"\n")
                    prob_files[f].write(str(particle[0])+"\n")
                    # Calculate component probabilities
                    comp_probs = np.zeros((particle[2]>0).sum())
                    comp_indexes = np.where(particle[2]>0)[0]
                    transition_cnet = self.transition_model.convert_to_cnet(evid_arr)
                    for c in xrange(comp_indexes.size):
                        evid_idx = np.array([False]*particle[2].size)
                        evid_idx[comp_indexes[c]] = True
                        comp_probs[c] = transition_cnet.prob_evid(sensor_cnet.tree,evid_idx,np.ones(1,dtype=int))
                    comp_files[f].write(','.join(map(str,comp_probs))+"\n")

                # Next, we need to get the particles we wish to send into the next iteration
                k_prev_particles = []
                for p in xrange(k):
                    if p == diff_indexes.size:
                        break
                    d = diff_indexes[-(p+1)]
                    k_prev_particles.append(new_particles[d])
                # DEBUG
                # print('k_prev_particles: '+str(k_prev_particles))
                # raw_input('Press any key to continue...')

        # Close all file handlers
        for f in xrange(len(map_files)):
            map_files[f].close()
            prob_files[f].close()
            comp_files[f].close()

        print('*** Finished Filtering!')

    def smooth_map(self, dataset_dir, results_dir, data_name, evid_percent="", k=1, l=1):
        # Initialization
        print('*** Smoothing dataset using Backward Algorithm')
        video_delims = read_video_file(dataset_dir, data_name) # where videos start and end in dataset
        evid_filename = dataset_dir + data_name + '.evidence' # where evidence dataset is stored
        map_filename = results_dir + data_name + '.' + str(k) + '.' + str(l) + '.predict' # where final map dataset is stored
        smooth_filename = results_dir + data_name + '.' + str(k) + '.' + str(l) + '.smooth'
        map_dataset = np.loadtxt(map_filename, dtype=int, delimiter=',')
        evid_dataset = np.loadtxt(evid_filename, dtype=int, delimiter=',')
        nrows, nvars = evid_dataset.shape[0], evid_dataset.shape[1]

        # Smooth all videos
        for v in xrange(video_delims.shape[0]-1,0,-1):
            print('** Processing Video',v+1,'of',video_delims.shape[0])
            start, end = video_delims[v-1], video_delims[v]-1
            evid_arr = evid_dataset[end:end+1]
            # Get sensor prob of last frame of video
            _, back_msg = self.sensor_model.map(evid_arr)
            map_dataset[end:end+1] = back_msg
            # Get transition probabilities for all other frames
            for e in xrange(end-1,start,-1):
                if e%500==0:
                  print('Started processing point',e+1)
                evid_arr = np.zeros((1,3*nvars),dtype=int)
                evid_arr[:,0:nvars] = map_dataset[e-1:e] # previous ground truth
                evid_arr[:,nvars:(2*nvars)] = back_msg # next ground truth
                evid_arr[:,(2*nvars):(3*nvars)] = evid_dataset[e:e+1] # sensor evidence
                back_msg = self.bitransition_model.map(evid_arr)[1]
                map_dataset[e:e+1] = back_msg

        # Write results to file
        np.savetxt(smooth_filename, map_dataset, dtype=int, delimiter=',', fmt='%i')
        print('*** Finished Smoothing!')

    def particle_filter(self, dataset_dir, results_dir, data_name, evid_percent="", m=100, sensor_only=False, notif=10):
        print('*** Particle Filtering (new)')
        print('m = ',m)
        if sensor_only : print('NOTE: Using only sensor model')
        video_delims = read_video_file(dataset_dir, data_name) # where videos start and end in dataset
        evid_filename = dataset_dir + data_name + '.evidence' # where evidence dataset is stored
        map_filename = results_dir + data_name + '.' + str(m) + '.predict' # where final map dataset is stored
        evid_dataset = np.loadtxt(evid_filename, dtype=int, delimiter=',')
        nrows, nvars = evid_dataset.shape[0], evid_dataset.shape[1]
        # Open map file
        with open(map_filename,"w") as map_file:
            # Start processing each video
            for v in xrange(video_delims.shape[0]-1):
                print('** Processing Video',v+1,'of',video_delims.shape[0])
                start, end = video_delims[v], video_delims[v+1]
                evid_arr = evid_dataset[start]
                particles = self.sensor_model.convert_to_cnet(evid_arr).generate_samples(m,sort_samples=True)
                for e in xrange(start+1,end):
                    if e%notif==0:
                        print('Started processing point',e+1)
                    map_idx, map_val = 0, -9999
                    for p in xrange(len(particles)):
                        # Evidence array
                        evid_arr = np.zeros(2*nvars,dtype=int)
                        evid_arr[0:nvars] = particles[p][2]
                        evid_arr[nvars:(2*nvars)] = evid_dataset[e:e+1][0]
                        # Sample new particles
                        particles[p] = self.transition_model.convert_to_cnet(evid_arr).generate_samples(1,sort_samples=True)[0]
                        particles[p] = (particles[p][0],p+1,particles[p][2])
                        if particles[p][0]>map_val:
                            map_idx = p
                            map_val = particles[p][0]
                    # Write results to file
                    map_file.write(','.join(map(str,particles[map_idx][2]))+"\n")
            print('*** Finished Particle Filtering (new)!')

    def computeLL(self, dataset_dir, results_dir, data_name, d_interval=1):
        print('*** Calculating Log-Likelihood of Data')
        # Initializations
        seq_delims = read_video_file(dataset_dir, data_name) # where sequences start and end in dataset
        evid_filename = dataset_dir + '/' + data_name + '.evidence' # where evidence dataset is stored
        # Load evidence
        evid_dataset = np.loadtxt(evid_filename, dtype=int, delimiter=',')
        nrows, nvars = evid_dataset.shape[0], self.transition_model.nvariables # Note: nvars is the number of variables in the MODEL
        nevids = evid_dataset.shape[1] - nvars
        llhood = np.zeros(seq_delims.shape[0]-1)
        # Initialize output files to be written to disk
        outfilename = results_dir + '/' + data_name + '.ll'
        # Open output file
        with open(outfilename,"w") as outfile:
            # Start processing each sequence partition
            for s in xrange(seq_delims.shape[0]-1):
                print('**** Processing Sequence',s+1,'of',seq_delims.shape[0]-1)
                start, end = seq_delims[s], seq_delims[s+1]
                # Calculate the likelihood of the first point w.r.t. sensor model
                if self.generative:
                    llhood[s] += np.log(self.sensor_model.prob_evid(self.sensor_model.tree, np.array([True] * nvars), evid_dataset[start]))
                else:
                    sensor_cnet = self.sensor_model.convert_to_cnet(evid_dataset[start,nvars:])
                    llhood[s] += np.log(sensor_cnet.prob_evid(sensor_cnet.tree, np.array([True] * nvars), evid_dataset[start,:nvars]))
                # For each point in the evidence for the current sequence, calculate LL score
                for e in xrange(start+1,end):
                    if e%d_interval==0:
                        print('Started processing point',e+1)
                    evid_arr = evid_dataset[e-1]
                    evid_arr[nvars:] = evid_dataset[e,nvars:]
                    curr_cnet = self.transition_model.convert_to_cnet(evid_arr)
                    llhood[s] += np.log(curr_cnet.prob_evid(curr_cnet.tree, np.array([True] * nvars), evid_dataset[e,:nvars]))
                llhood[s] /= (end-start)
                outfile.write(str(s+1) + ": " + str(llhood[s]) + '\n')
            # Print the average of all llhoods
            outfile.write("Total: " + str(np.mean(llhood)))

def to_str(idx_arr, arr):
    r = ''
    idx_only = np.where(idx_arr == True)[0]
    for i in xrange(arr.size):
        r += 'X' + str(idx_only[i]) + '=' + str(arr[i]) + ', '

    return r[:-2]

def compare_dists_dataset(p, q, dir, filename, results_dir = '', p_name = 'p', q_name = 'q', smooth=False):
    test_filename = dir + filename
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    if test_dataset.shape[1] != q.nvariables or (p.nvariables != q.nvariables and p.nvariables != 2*q.nvariables):
        return np.array([])
    numsamples = test_dataset.shape[0]
    error, p1, p2 = np.zeros(numsamples), np.zeros(numsamples), np.zeros(numsamples)
    evid_idx_p = np.array([True] * p.nvariables)
    evid_idx_q = np.array([True] * q.nvariables)
    if not smooth:
        evid_idx_p[q.nvariables:] = False
    else:
        evid_idx_p[:q.nvariables] = False
    for d in xrange(numsamples):
        p1[d] = p.prob_evid(p.tree, evid_idx_p, test_dataset[d, :])
        p2[d] = q.prob_evid(q.tree, evid_idx_q, test_dataset[d, :])
        error[d] = np.abs(p1[d] - p2[d])
    results = np.hstack((test_dataset, np.vstack((p1, p2, error)).T))
    if len(results_dir) > 0:
        format = '%i,' * q.nvariables + '%.4f,%.4f,%.4f'
        outfilename = results_dir + p_name + '.' + q_name + '.' + filename
        np.savetxt(outfilename, results, fmt=format)
        file = open(outfilename, 'a')
        file.write('Average Error: ' + str(error.mean()))
        file.close()
    return results

def verify_posterior(dataset_dir, data_name, max_depth, evid_percent, function_type, alpha, partition, samples, smooth=False):
    # Learn CCNET
    print('** Learning CCNET for transition distribution')
    p_ccnet = CNET_CLT([], max_depth)
    filename = dataset_dir + data_name + '.transition'
    dataset = np.loadtxt(filename, dtype=int, delimiter=',')
    evid_var = read_evidence_file(dataset_dir, evid_percent, data_name + '.transition')
    query_var = np.setdiff1d(np.arange(dataset.shape[1]), evid_var)
    evid_arr = dataset[:, evid_var]
    query_arr = dataset[:, query_var]
    p_ccnet.learnStructure(evid_arr, query_arr, function_type, alpha)
    # Load evidence
    evid_filename = dataset_dir + data_name + '.' + str(partition) + '.evidence'
    evid_dataset = np.loadtxt(evid_filename, dtype=int, delimiter=',')
    # Initialize Parameters
    start = 1 if not smooth else evid_dataset.shape[0]-2
    end = evid_dataset.shape[0]-1 if not smooth else -1
    # DEBUG
    end = 233
    inc = 1 if not smooth else -1
    suffix = 'f' if not smooth else 's'
    lahead = 0 if not smooth else 1
    pmsg = -1 if not smooth else 1
    q_prev, q_curr = CNET.CNET(), CNET.CNET()
    nvars = p_ccnet.nvariables / 2
    # Iterate over evidence
    for e in xrange(start, end, inc):
        outfilename = str(partition) + '.' + suffix + '.p' + str(e+1) + '.q.update.txt'
        prev_model_name = data_name+'.'+str(partition)+'.'+suffix+'.'+str(e+1+pmsg)
        curr_model_name = data_name+'.'+str(partition)+'.'+suffix+'.'+str(e+1)
        print('*** Processing Point '+str(e+1))
        print('Evidence = '+str(e+1+lahead))
        print('Prev Message = '+str(e+1+pmsg))
        print('Prev Model Loaded = '+prev_model_name)
        p_cnet = p_ccnet.convert_to_cnet(evid_dataset[e])
        q_prev.load(dataset_dir+'/models/',prev_model_name)
        q_curr.load(dataset_dir+'/models/',curr_model_name)
        sample_prob = np.zeros((samples.shape[0],2))
        for s in xrange(samples.shape[0]):
            # Set query parameters
            marg_idx = np.array(([True]*nvars)+([False]*nvars)) if not smooth else np.array(([False]*nvars)+([True]*nvars))
            marg = samples[s]
            evid_idx = np.array(([False]*nvars)+[True]+([False]*(nvars-1))) if not smooth else np.array([True]+[False]*((2*nvars)-1))
            evid0 = np.array([0])
            evid1 = np.array([1])
            # Calculate exact update probabilities
            psample_0 = p_cnet.prob_marg_evid(evid_idx, evid0, marg_idx, marg)
            psample_1 = p_cnet.prob_marg_evid(evid_idx, evid1, marg_idx, marg)
            q0 = q_prev.prob_evid(q_prev.tree, evid_idx[nvars:], evid0) if not smooth else q_prev.prob_evid(q_prev.tree, evid_idx[:nvars], evid0)
            q1 = 1-q0
            sample_prob[s,0] = (psample_0*q0) + (psample_0*q1)
            # Calculate model computed probabilities
            qsample_curr = q_curr.prob_evid(q_curr.tree,np.array([True]*nvars),marg)
            sample_prob[s,1] = qsample_curr
        # DEBUG
        print((sample_prob[:,0] - sample_prob[:,1]).mean())
        # Write to file
        format = ('%i,' * nvars) + '%.4f,%.4f'
        np.savetxt(dataset_dir+'/samples/'+outfilename,np.hstack((samples,sample_prob)),fmt=format)

# For the time being, only filtering
def filter_smooth_map(model_dir,results_dir,data_name,partition,start=1,end=0):
    if start > end:
        fsize = len(glob.glob(model_dir+'/'+data_name+'.'+str(partition)+'.f.*.model'))
        ssize = len(glob.glob(model_dir+'/'+data_name+'.'+str(partition)+'.s.*.model'))
        # if fsize != ssize: return
        start = 1
        end = fsize+1
    map_filename = data_name + '.' + str(partition) + '.ep.predict'
    with open(results_dir+'/'+map_filename,'w') as map_file:
        for t in xrange(start,end):
            print('Processing time slice '+str(t))
            filter_filename = data_name + '.' + str(partition) + '.f.' + str(t)
            smooth_filename = data_name + '.' + str(partition) + '.s.' + str(t)
            q_filter, q_smooth = CNET.CNET(), CNET.CNET()
            q_filter.load(model_dir,filter_filename)
            # q_smooth.load(model_dir,smooth_filename)
            # q_filter.project_and_update(q_smooth,q_smooth,True)
            map_tuple = q_filter.map()[1]
            print('[DEBUG] map_tuple: '+str(map_tuple))
            map_file.write(','.join(map(str, map_tuple))+'\n')

def read_evidence_file(file_dir, evid_percent, data_name):
    input =  open(file_dir + '/evidence'+ evid_percent + '.txt')
    in_lines =  input.readlines()
    input.close()

    total_datasets = len(in_lines) / 2
    for i in xrange(total_datasets):
        if in_lines[2*i].strip() == data_name:
            evid = in_lines[2*i+1].strip().split(',')
            evid[0] = evid[0][1:]
            evid[-1] = evid[-1][:-1]

    #print (evid)
    evid_arr = np.array(evid).astype(np.int)
    #print (evid_arr)

    return evid_arr

'''
    Chiro: This file tells the system how the videos are divided
'''
def read_video_file(file_dir, data_name):
    input =  open(file_dir + '/video.txt')
    in_lines =  input.readlines()
    input.close()

    total_datasets = len(in_lines) / 2
    for i in xrange(total_datasets):
        if in_lines[2*i].strip() == data_name:
            vid = in_lines[2*i+1].strip().split(',')
            vid[0] = vid[0][1:]
            vid[-1] = vid[-1][:-1]

    #print (evid)
    vid_arr = np.array(vid).astype(np.int)
    #print (evid_arr)

    return vid_arr

'''
    Chiro: Method to calculate K-scores, Jaccard index and Hamming Loss
'''
def calc_statistics(dataset_dir, results_dir, data_name, suffix_name, k, l):
    print('*** Calculating Statistics')

    # Read in datasets
    ground_filename = dataset_dir + data_name + '.ground'
    predict_filename = results_dir + data_name + '.' + str(k) + '.' + str(l) + '.' + suffix_name
    ground_dataset = np.loadtxt(ground_filename, dtype=int, delimiter=',')
    predict_dataset = np.loadtxt(predict_filename, dtype=int, delimiter=',')
    correct_dataset = ground_dataset * predict_dataset

    # Prepare label statistics
    ground_labels = np.sum(ground_dataset, axis=1)
    predict_labels = np.sum(predict_dataset, axis=1)
    correct_labels = np.sum(correct_dataset, axis=1)
    all_labels = (ground_labels + predict_labels) - correct_labels

    # Calculate K-scores
    all_correct_labels = ground_labels - correct_labels
    all_correct_indices = np.where(all_correct_labels==0)[0]
    k1_score = np.count_nonzero(correct_labels) / (correct_labels.shape[0]*1.0)
    k2_score = np.union1d(np.where(correct_labels>=2)[0],all_correct_indices).shape[0] / (correct_labels.shape[0]*1.0)
    k3_score = np.union1d(np.where(correct_labels>=3)[0],all_correct_indices).shape[0] / (correct_labels.shape[0]*1.0)
    k4_score = np.union1d(np.where(correct_labels>=4)[0],all_correct_indices).shape[0] / (correct_labels.shape[0]*1.0)

    # Calculate Jaccard Index
    jaccard_index = np.sum( (correct_labels * 1.0) / all_labels ) / ground_labels.shape[0]

    # Calculate Hamming Loss
    hamming_loss = np.sum( ( (ground_labels - correct_labels) * 1.0) / ground_labels ) / ground_labels.shape[0]

    # Print statistics
    print('K-1 Score:',k1_score)
    print('K-2 Score:',k2_score)
    print('K-3 Score:',k3_score)
    print('K-4 Score:',k4_score)
    print('Jaccard Index:',jaccard_index)
    print('Hamming Loss:',hamming_loss)

def main_ccnet():

    dataset_dir = sys.argv[2]
    results_dir = sys.argv[4]
    data_name = sys.argv[6]
    max_depth = int(sys.argv[8])
    evid_percent = sys.argv[10]
    function_type = sys.argv[12]  # LR or NN
    alpha = float(sys.argv[14])  # the NN regulization coef

#
#    test_arr = np.ones(10, dtype = int).reshape(10,1)
#    print (test_arr)
#    test_func = count_func(test_arr)
#    print (test_func.predict_log_proba())
#    ssss

    print('------------------------------------------------------------------')
    print('Conditional Cutset Network')
    print('------------------------------------------------------------------')


    #train_filename = sys.argv[1]
    train_filename = dataset_dir + data_name + '.train.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'

    #out_file = '../module/' + data_name + '.npz'
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    print ("********* Using Validation Dataset in Training ************")
    train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    #print("Learning Chow-Liu Trees on original data ......")
    #clt = CLT()
    #clt.learnStructure(train_dataset)
    #print("done")

    #train_dataset = train_dataset[:50,:]

    n_variables = train_dataset.shape[1]

    ### Load the trained mixture of clt

    # evidence file

    evid_var = read_evidence_file(dataset_dir, evid_percent, data_name)
    query_var =  np.setdiff1d(np.arange(n_variables), evid_var)
    evid_arr = train_dataset[:,evid_var]
    query_arr = train_dataset[:, query_var]
    print ('evid_var: ', evid_var)
    print ('query_var: ', query_var)


    print("Learning Conditional Cutset Networks from data .....")
    #output_dir = '../ccnet_results/' + data_name + '/'
    #for lamda in lamda_array:
    #print ("Current Lamda: ", lamda)
    #n_variable = valid_dataset.shape[1]
    #cnets = []
    tree = []
    #max_depth = 11
    #output_cnet = '../cnet_module/'
    out_file = results_dir + data_name + '_' + evid_percent  + '_' + function_type + '_' + str(alpha)+'.txt'
    #test_out_file = output_dir + 'test.txt'

    train_ll_score = 0.0
    valid_ll_score = 0.0
    test_ll_score = 0.0

    cnet  = CNET_CLT(tree, depth = max_depth)
    cnet.learnStructure(evid_arr, query_arr, function_type, alpha)
    #cnets.append(cnet)
    #tree = copy.deepcopy(cnet.tree)

    # compute ll score
    train_ll_score = cnet.computeLL(train_dataset[:,evid_var], train_dataset[:,query_var]) / train_dataset.shape[0]
    print('Train set cnet LL scores:', train_ll_score)
    valid_ll_score = cnet.computeLL(valid_dataset[:,evid_var], valid_dataset[:,query_var]) / valid_dataset.shape[0]
    print('Valid set cnet LL scores:', valid_ll_score)
    test_ll_score = cnet.computeLL(test_dataset[:,evid_var], test_dataset[:,query_var]) / test_dataset.shape[0]
    print('Test set cnet LL scores:', test_ll_score)
    #print (train_ll_score[i-1])
    #print (valid_ll_score[i-1])
    #print (test_ll_score[i-1])
    #print ()
    with open(out_file, 'w') as f_handle:
        np.savetxt(f_handle, np.array([train_ll_score, valid_ll_score, test_ll_score]), delimiter=',')


    # save cnet module to file

    #print ("save module: ", i)
    #main_dict = {}
    #utilM.save_cutset(main_dict, cnet.tree, np.arange(n_variables), ccpt_flag = True)
    #np.save(output_cnet + data_name + '_' + str(i), main_dict)
    #np.savez_compressed(output_cnet + data_name + str(i), module = main_dict)

    print("done")





"""
Learnt mixture of CLT
"""
def main_clt():
    #train_filename = sys.argv[1]


    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    seq = sys.argv[6]
    n_components = int(sys.argv[8])
    max_iter = int(sys.argv[10])
    epsilon = float(sys.argv[12])


    train_name = dataset_dir + data_name +'.train.data'
    valid_name = dataset_dir + data_name +'.valid.data'
    test_name = dataset_dir + data_name +'.test.data'
    data_train = np.loadtxt(train_name, delimiter=',', dtype=np.uint32)
    data_valid = np.loadtxt(valid_name, delimiter=',', dtype=np.uint32)
    data_test = np.loadtxt(test_name, delimiter=',', dtype=np.uint32)

    #epsilon *= data_train.shape[0]

    print("Learning Mixture of Chow-Liu Trees on original data ......")
    #n_components = 5
    mix_clt = MIXTURE_CLT()
    mix_clt.learnStructure(data_train, n_components)
    mix_clt.EM(data_train, max_iter, epsilon)

    #print("done")

    save_list = []
    #save_dict['weights'] = mix_clt.mixture_weight
    #print (save_dict['weights'])
    for i in xrange(n_components):
        new_dict = dict()
        new_dict['xprob'] = mix_clt.clt_list[i].xprob
        new_dict['xyprob'] = mix_clt.clt_list[i].xyprob
        new_dict['topo_order'] = mix_clt.clt_list[i].topo_order
        new_dict['parents'] = mix_clt.clt_list[i].parents
        new_dict['log_cond_cpt'] = mix_clt.clt_list[i].log_cond_cpt
        new_dict['tree'] = mix_clt.clt_list[i].Tree
        save_list.append(new_dict)


    valid_ll = mix_clt.computeLL(data_valid) / data_valid.shape[0]
    test_ll = mix_clt.computeLL(data_test) / data_test.shape[0]

    out_file = '../output/' + data_name +'_'+str(seq)+'_'+str(n_components) +'.npz'

    #np.savez('save_dict.npz', save_dict)
    np.savez_compressed(out_file, clt_component=save_list, weights=mix_clt.mixture_weight, valid_ll = valid_ll, test_ll = test_ll)


    print('Test set LL scores')
    print(test_ll, "Mixture-Chow-Liu")



    print('Valid set LL scores')
    print(valid_ll, "Mixture-Chow-Liu")




def main_ccnet_hmm():

    print('------------------------------------------------------------------')
    print('Hidden Markov Models Using Conditional Cutset Networks')
    print('------------------------------------------------------------------')

    # Read in arguments
    dataset_dir = sys.argv[2]
    results_dir = sys.argv[4]
    data_name = sys.argv[6]
    max_depth = int(sys.argv[8])
    evid_percent = sys.argv[10]
    function_type = sys.argv[12]  # LR or NN
    alpha = float(sys.argv[14])  # the NN regulization coef
    k = 2 # Value for kl-particle filtering
    l = 3
    m = 100 # Value for kl-particle filtering
    if(len(sys.argv)>=17): k = int(sys.argv[16])
    if(len(sys.argv)==19): l = int(sys.argv[18])

    # Create and train a HMM_CNET
    hmm = HMM_CNET(max_depth)
    hmm.train(dataset_dir,data_name,evid_percent,function_type,alpha)

    # Calculate Filtering Results
    hmm.filter_map(dataset_dir,results_dir,data_name,evid_percent,k,l,m,sensor_only=True)
    calc_statistics(dataset_dir,results_dir,data_name,'predict',k,m)

    # Calculate Smoothing Results
    # hmm.smooth_map(dataset_dir,results_dir,data_name,evid_percent,k,l)
    # calc_statistics(dataset_dir,results_dir,data_name,'smooth',k,l)

    print("FINISHED PROGRAM")

def main_ccnet_dccn():

    print('------------------------------------------------------------------')
    print('Dynamic Conditional Cutset Networks')
    print('------------------------------------------------------------------')

    # Define usage string
    usage_string = "Usage:\n" + \
                    "python ccnet.py <data-dir> <results-dir> <data-name> <max-depth> <evid-percent> <func> <alpha>\n" + \
                    "  [ -p <partition-no> -alg <algorithm> -struct <structure> -cltlearn <interval> -save -samples -restore ]\n\n" + \
                    "Positional Arguments:\n" + \
                    "1.\t<data-dir>\t\t:\tData Directory (contains actual data, evidence file, etc.)\n" + \
                    "2.\t<results-dir>\t\t:\tResults Directory (output files generated here)\n" + \
                    "3.\t<data-name>\t\t:\tData Name (name of the dataset)\n" + \
                    "4.\t<max-depth>\t\t:\tMaximum Depth (of the cutset network)\n" + \
                    "5.\t<evid-percent>\t\t:\tEvidence Percentage (how much of the evidence should be used for training)\n" + \
                    "6.\t<func>\t\t\t:\tFunction (for P(Y|X) density function; either LR or NN)\n" + \
                    "7.\t<alpha>\t\t\t:\tAlpha parameter for learning conditional function\n\n" + \
                    "Optional Arguments and Flags:\n" + \
                    "1.\t-p <partition>\t\t:\tPartition number of test set [ Default: -1 ]\n" + \
                    "2.\t-alg <algorithm>\t:\tAlgorithm used for filtering (ep/pf)  [ Default: ep ]\n" + \
                    "\t\t\t\t:\tep - Expectation Propagation\n" + \
                    "\t\t\t\t:\tpf - Particle Filtering\n" + \
                    "3.\t-struct <structure>\t:\t(EP) Structure used for Expectation Propagation [ Default: 0 ]\n" + \
                    "\t\t\t\t:\t0 - Prior structure\n" + \
                    "\t\t\t\t:\t1 - Skewed posterior structure\n" + \
                    "4.\t-cltlearn <interval>\t:\t(EP) Specifies how frequently structure learning should be done [ Default: 0 ]\n" + \
                    "5.\t-k <num-particles>\t:\t(PF) Number of particles for particle filtering [ Default: 1 ]\n" + \
                    "6.\t-save\t\t\t:\tSaves model at each time slice to models directory [ Default: OFF ]\n" + \
                    "7.\t-samples\t\t:\tCompares likelihoods of samples in all.samples.txt [ Default: OFF ]\n" + \
                    "8.\t-restore\t\t:\tSearches models directory for models and restores them [ Default: OFF ]\n\n" + \
                    "Example:\n" + \
                    "python ccnet.py data/nips/synthetic/exp1 data/nips/synthetic/exp1 exp1 14 "" LR 0.1 -p 2 -alg pf -k 10 -samples\n"

    # Check if all arguments are present
    if len(sys.argv) < 8:
        print(usage_string)
        return

    # Read in arguments
    dataset_dir = sys.argv[1] + "/"
    results_dir = sys.argv[2] + "/"
    data_name = sys.argv[3]
    max_depth = int(sys.argv[4])
    evid_percent = sys.argv[5]
    function_type = sys.argv[6]  # LR or NN
    alpha = float(sys.argv[7])  # the NN regulization coef
    partition = -1
    alg = "ep"
    struct = 0
    cltlearn = 0
    k = 1
    samples_flag = False
    save_flag = False
    restore_flag = False

    opt_args = sys.argv[-(len(sys.argv) - 8):]
    prev_option = 'none'
    for a in xrange(0,len(opt_args)+1):
        # Check single options
        if prev_option == '-samples':
            samples_flag = True
            prev_option = 'none'
        elif prev_option == '-save':
            save_flag = True
            prev_option = 'none'
        elif prev_option == '-restore':
            restore_flag = True
            prev_option = 'none'
        # Check previous option
        if prev_option=='none' and a<len(opt_args):
            prev_option = opt_args[a].strip()
            continue
        # Check double options
        if prev_option == '-p':
            partition = int(opt_args[a])
        elif prev_option == '-alg':
            alg = opt_args[a]
        elif prev_option == '-struct':
            struct = int(opt_args[a])
        elif prev_option == '-cltlearn':
            cltlearn = int(opt_args[a])
        elif prev_option == '-k' and int(opt_args[a])>1:
            k = int(opt_args[a])
        prev_option = 'none'

    print('*** Initialization Arguments')
    print('dataset_dir = '+dataset_dir)
    print('results_dir = '+results_dir)
    print('data_name = '+data_name)
    print('max_depth = '+str(max_depth))
    print('evid_percent = '+evid_percent)
    print('function_type = '+function_type)
    print('alpha = '+str(alpha))
    print('partition = '+str(partition))
    print('alg = '+alg)
    print('struct = '+str(struct))
    print('cltlearn = '+str(cltlearn))
    print('k = '+str(k))
    print('samples = '+str(samples_flag))
    print('save = '+str(save_flag))
    print('restore = '+str(restore_flag))

    # Create and train a DCCN
    dccn = DCCN(max_depth)
    dccn.train(dataset_dir, data_name, evid_percent, function_type, alpha, \
                restore=restore_flag, save=save_flag)

    # Calculate Filtering Results
    if alg=="ep":
        dccn.filter_map(dataset_dir,results_dir,data_name,evid_percent,partition, \
                struct,save_flag,samples_flag,restore_flag,cltlearn=cltlearn)
    else:
        dccn.particle_filter(dataset_dir,results_dir,data_name,evid_percent,partition,k)
    # calc_statistics(dataset_dir,results_dir,data_name,'predict',k,l)

    # Calculate Smoothing Results
    # hmm.smooth_map(dataset_dir,results_dir,data_name,evid_percent,k,l)
    # calc_statistics(dataset_dir,results_dir,data_name,'smooth',k,l)

    print("FINISHED PROGRAM")

if __name__=="__main__":

    start = time.time()
    # Chiro
    # main_ccnet()
    main_ccnet_hmm()
    # main_ccnet_dccn()
    print ('Total running time: ', time.time() - start)
