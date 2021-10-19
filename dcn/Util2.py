import numpy as np
import time

SMOOTH = 0.00000000000000000000000000000000001

class Util2(object):
    def __init__(self):
        self.times=0

    @staticmethod
    def str_dict(d):
        r = ""
        for key in d.keys():
            r += str(key) + ": " + str(d[key]) + "\n"
        return r

    @staticmethod
    def compute_xycounts_slow(dataset,timing=False):
        start = time.time()
        nvariables=dataset.shape[1]
        prob_xy = np.zeros((nvariables, nvariables, 2, 2))
        for i in range(nvariables):
            for j in range(nvariables):
                prob_xy[i][j][0][0] = np.count_nonzero((dataset[:, i] == 0) & (dataset[:, j] == 0))
                prob_xy[i][j][0][1] = np.count_nonzero((dataset[:, i] == 0) & (dataset[:, j] == 1))
                prob_xy[i][j][1][0] = np.count_nonzero((dataset[:, i] == 1) & (dataset[:, j] == 0))
                prob_xy[i][j][1][1] = np.count_nonzero((dataset[:, i] == 1) & (dataset[:, j] == 1))
        if timing:
            print (time.time()-start)
        return prob_xy

    @staticmethod
    def compute_xcounts_slow(dataset,timing=False):
        start = time.time()
        nvariables = dataset.shape[1]
        prob_x = np.zeros((nvariables, 2))
        for i in range(nvariables):
            prob_x[i][0] = np.count_nonzero(dataset[:, i] == 0)
            prob_x[i][1] = dataset.shape[0]-prob_x[i][0]
        if timing:
            print(time.time() - start)
        return prob_x

    @staticmethod
    def compute_xycounts(dataset,timing=False):
        start = time.time()
        nvariables=dataset.shape[1]
        prob_xy = np.zeros((nvariables, nvariables, 2, 2))
        #print np.nonzero(dataset)
        #print np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 0).astype(int))#
        prob_xy[:, :, 0, 0] = np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 0).astype(int))
        prob_xy[:, :, 0, 1] = np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 1).astype(int))
        prob_xy[:, :, 1, 0] = np.einsum('ij,ik->jk', (dataset == 1).astype(int), (dataset == 0).astype(int))
        prob_xy[:, :, 1, 1] = np.einsum('ij,ik->jk', (dataset == 1).astype(int), (dataset == 1).astype(int))
        if timing:
            print(time.time() - start)
        return prob_xy

    @staticmethod
    def compute_xcounts(dataset,timing=False):
        start = time.time()
        nvariables = dataset.shape[1]
        prob_x = np.zeros((nvariables, 2))
        prob_x[:,0]=np.einsum('ij->j',(dataset == 0).astype(int))
        prob_x[:,1] = np.einsum('ij->j',(dataset == 1).astype(int))
        if timing:
            print(time.time() - start)
        return prob_x

    # Shasha's code
    @staticmethod
    def compute_weighted_xycounts(dataset,weights, timing=False):
        start = time.time()
        nvariables=dataset.shape[1]
        prob_xy = np.zeros((nvariables, nvariables, 2, 2))
        #print np.nonzero(dataset)
        #print np.einsum('ij,ik->jk', (dataset == 0).astype(int), (dataset == 0).astype(int))#
        prob_xy[:, :, 0, 0] = np.einsum('ij,ik->jk', (dataset == 0).astype(int)* weights[:, np.newaxis], (dataset == 0).astype(int))
        prob_xy[:, :, 0, 1] = np.einsum('ij,ik->jk', (dataset == 0).astype(int)* weights[:, np.newaxis], (dataset == 1).astype(int))
        prob_xy[:, :, 1, 0] = np.einsum('ij,ik->jk', (dataset == 1).astype(int)* weights[:, np.newaxis], (dataset == 0).astype(int))
        prob_xy[:, :, 1, 1] = np.einsum('ij,ik->jk', (dataset == 1).astype(int)* weights[:, np.newaxis], (dataset == 1).astype(int))
        if timing:
            print(time.time() - start)

        return prob_xy

    # Shasha's code
    @staticmethod
    def compute_weighted_xcounts(dataset,weights, timing=False):
        start = time.time()
        nvariables = dataset.shape[1]
        prob_x = np.zeros((nvariables, 2))
        prob_x[:,0]=np.einsum('ij->j',(dataset == 0).astype(int) * weights[:, np.newaxis])
        prob_x[:,1] = np.einsum('ij->j',(dataset == 1).astype(int) * weights[:, np.newaxis])
        if timing:
            print(time.time() - start)
        return prob_x

    # Chiro's code
    @staticmethod
    def compute_xycounts_new(dset, select_idx=np.array([]), weights=np.array([]), timing=False):
        start = time.time()
        nvars = dset.shape[1]
        all_idx = np.arange(nvars)
        select_idx = select_idx if (np.setdiff1d(all_idx, select_idx).size == 0) else np.arange(nvars)
        weights = weights.reshape(dset.shape[0], 1) if () else np.ones((dset.shape[0], 1))
        prob_xy = np.zeros((nvars, select_idx.size, 2, 2))
        prob_xy[:, :, 0, 0] = np.einsum('ij,ik->jk', (dset == 0) * weights, (dset[:, select_idx] == 0).astype(int))
        prob_xy[:, :, 0, 1] = np.einsum('ij,ik->jk', (dset == 0) * weights, (dset[:, select_idx] == 1).astype(int))
        prob_xy[:, :, 1, 0] = np.einsum('ij,ik->jk', (dset == 1) * weights, (dset[:, select_idx] == 0).astype(int))
        prob_xy[:, :, 1, 1] = np.einsum('ij,ik->jk', (dset == 1) * weights, (dset[:, select_idx] == 1).astype(int))
        if timing:
            print(time.time() - start)
        return prob_xy

    @staticmethod
    def compute_xcounts_new(dset, select_idx=np.array([]), weights=np.array([]), timing=False):
        start = time.time()
        nvars = dset.shape[1]
        all_idx = np.arange(nvars)
        select_idx = select_idx if (np.setdiff1d(all_idx, select_idx).size == 0) else np.arange(nvars)
        weights = weights.reshape(dset.shape[0], 1) if () else np.ones((dset.shape[0], 1))
        prob_x = np.zeros((select_idx.size, 2))
        prob_x[:, 0] = np.einsum('ij->j', (dset[:, select_idx] == 0) * weights)
        prob_x[:, 1] = np.einsum('ij->j', (dset[:, select_idx] == 1) * weights)
        if timing:
            print(time.time() - start)
        return prob_x

    @staticmethod
    def compute_edge_weights_new(xycounts, xcounts, ycounts, timing=False):
        start = time.time()
        p_xy = Util2.normalize2d(xycounts)
        # print p_xy
        p_x_r = np.reciprocal(Util2.normalize1d(xcounts))
        p_y_r = np.reciprocal(Util2.normalize1d(ycounts))

        # print p_xy[7]
        # print 1.0/p_x_r[7]
        # sum_xy_fast=np.einsum('ijkl,ijkl->ij',p_xy,np.log(p_xy))+np.einsum('ijkl,ik->ij',p_xy,np.log(p_x_r))+np.einsum('ijkl,jl->ij',p_xy,np.log(p_x_r))
        sum_xy = np.zeros((p_x_r.shape[0], p_y_r.shape[0]))
        sum_xy += p_xy[:, :, 0, 0] * np.log(np.einsum('ij,i,j->ij', p_xy[:, :, 0, 0], p_x_r[:, 0], p_y_r[:, 0]))
        sum_xy += p_xy[:, :, 0, 1] * np.log(np.einsum('ij,i,j->ij', p_xy[:, :, 0, 1], p_x_r[:, 0], p_y_r[:, 1]))
        sum_xy += p_xy[:, :, 1, 0] * np.log(np.einsum('ij,i,j->ij', p_xy[:, :, 1, 0], p_x_r[:, 1], p_y_r[:, 0]))
        sum_xy += p_xy[:, :, 1, 1] * np.log(np.einsum('ij,i,j->ij', p_xy[:, :, 1, 1], p_x_r[:, 1], p_y_r[:, 1]))
        if timing:
            print(time.time() - start)
            print sum_xy
            # print sum_xy_fast
        # print "from count:"
        # print sum_xy[7]
        return sum_xy

    @staticmethod
    def normalize2d(xycounts):
        xycountsf=xycounts.astype(np.float64)
        norm_const=np.einsum('ijkl->ij',xycountsf)
        return xycountsf/norm_const[:,:,np.newaxis,np.newaxis]

    @staticmethod
    def normalize1d(xcounts):
        xcountsf = xcounts.astype(np.float64)
        norm_const = np.einsum('ij->i', xcountsf)
        return xcountsf/norm_const[:,np.newaxis]

    @staticmethod
    def normalize(weights):
        norm_const=np.sum(weights)
        return weights/norm_const

    # Shasha's code
    @staticmethod
    def normalize1d_in_2d(xycounts):
        xycountsf=xycounts.astype(np.float64)
        norm_const=np.einsum('ijk->i',xycountsf)
        return xycountsf/norm_const[:,np.newaxis,np.newaxis]

    @staticmethod
    # normalize the matirx for each columns, and compute ll score
    # input weights are in log form
    # return normalized weights, Not in log form and ll score
    def m_step_trick(log_weights):
        #print log_weights
        max_arr = np.max(log_weights, axis = 0)
        #print "max: "
        #print max_arr

        weights = np.exp(log_weights - max_arr[np.newaxis,:])
        norm_const = np.einsum('ij->j', weights)
        #print "norm:"
        #print norm_const
        weights = weights / norm_const[np.newaxis,:]
        #print " normalized weights: ", weights

        ll_score = np.sum(np.log(norm_const)) + np.sum(max_arr)

        """
        exp_weights = np.exp(log_weights)
        temp_ll = 0.0
        for i in xrange(exp_weights.shape[1]):
            temp_ll += np.log(np.sum(exp_weights[:,i]))

        print "ll score: ", ll_score, temp_ll
        """
        return weights, ll_score


    @staticmethod
    # normalize the matirx for each columns, and compute ll score
    # input weights are in log form
    # return normalized weights, Not in log form and ll score
    def get_ll_trick(log_weights):
        #print log_weights
        max_arr = np.max(log_weights, axis = 0)
        #print "max: "
        #print max_arr

        weights = np.exp(log_weights - max_arr[np.newaxis,:])
        norm_const = np.einsum('ij->j', weights)

        ll_scores = np.log(norm_const) + max_arr


        return ll_scores

    @staticmethod
    def compute_edge_weights_slow(xycounts, xcounts,timing=False):
        start = time.time()
        p_xy=Util2.normalize2d(xycounts)
        p_x=Util2.normalize1d(xcounts)
        log_px = np.log(p_x)
        log_pxy = np.log(p_xy)
        nvariables=p_x.shape[0]
        sum_xy = np.zeros((nvariables, nvariables))
        for i in range(nvariables):
            for j in range(nvariables):
                sum_xy[i][j] += p_xy[i][j][0][0] * (log_pxy[i][j][0][0] - log_px[i][0] - log_px[j][0])
                sum_xy[i][j] += p_xy[i][j][0][1] * (log_pxy[i][j][0][1] - log_px[i][0] - log_px[j][1])
                sum_xy[i][j] += p_xy[i][j][1][0] * (log_pxy[i][j][1][0] - log_px[i][1] - log_px[j][0])
                sum_xy[i][j] += p_xy[i][j][1][1] * (log_pxy[i][j][1][1] - log_px[i][1] - log_px[j][1])
        if timing:
            print(time.time() - start)
        return sum_xy

    @staticmethod
    def compute_edge_weights(xycounts,xcounts,timing=False):
        start = time.time()
        p_xy = Util2.normalize2d(xycounts)
        #print p_xy
        p_x_r = np.reciprocal(Util2.normalize1d(xcounts))

        #print p_xy[7]
        #print 1.0/p_x_r[7]
        #sum_xy_fast=np.einsum('ijkl,ijkl->ij',p_xy,np.log(p_xy))+np.einsum('ijkl,ik->ij',p_xy,np.log(p_x_r))+np.einsum('ijkl,jl->ij',p_xy,np.log(p_x_r))
        sum_xy=np.zeros((p_x_r.shape[0], p_x_r.shape[0]))
        sum_xy += p_xy[:,:,0,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,0],p_x_r[:,0],p_x_r[:,0]))
        sum_xy += p_xy[:,:,0,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,1],p_x_r[:,0],p_x_r[:,1]))
        sum_xy += p_xy[:,:,1,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,0],p_x_r[:,1],p_x_r[:,0]))
        sum_xy += p_xy[:,:,1,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,1],p_x_r[:,1],p_x_r[:,1]))
        if timing:
            print(time.time() - start)
            print sum_xy
            #print sum_xy_fast
        #print "from count:"
        #print sum_xy[7]
        return sum_xy


    @staticmethod
    # Shasha's code, basically the same as compute_edge_weights
    # The only difference is the input in probablity, not count
    def compute_MI_prob(p_xy,p_x,timing=False):
        start = time.time()
        p_x_r = np.reciprocal(p_x)

        #sum_xy_fast=np.einsum('ijkl,ijkl->ij',p_xy,np.log(p_xy))+np.einsum('ijkl,ik->ij',p_xy,np.log(p_x_r))+np.einsum('ijkl,jl->ij',p_xy,np.log(p_x_r))
        sum_xy=np.zeros((p_x_r.shape[0], p_x_r.shape[0]))
        sum_xy += p_xy[:,:,0,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,0],p_x_r[:,0],p_x_r[:,0]))
        sum_xy += p_xy[:,:,0,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,0,1],p_x_r[:,0],p_x_r[:,1]))
        sum_xy += p_xy[:,:,1,0]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,0],p_x_r[:,1],p_x_r[:,0]))
        sum_xy += p_xy[:,:,1,1]*np.log(np.einsum('ij,i,j->ij',p_xy[:,:,1,1],p_x_r[:,1],p_x_r[:,1]))
        if timing:
            print(time.time() - start)
            print sum_xy
            #print sum_xy_fast
        #print "from prob:"
        #print sum_xy[7]
        return sum_xy

    # Shasha's code
    @staticmethod
    def compute_conditional_CPT(xyprob,xprob,topo_order, parents, timing=False):

        #print "topo_order: ", topo_order
        start = time.time()
        nvariables = xprob.shape[0]
        cond_cpt = np.zeros((nvariables,2,2))

        # for the root we have a redundant representation
        root = topo_order[0]
        cond_cpt[0, 0, :] = xprob[root, 0]
        cond_cpt[0, 1, :] = xprob[root, 1]




        for i in xrange(1, nvariables):
            x = topo_order[i]
            y = parents[x]

            # id, child, parent

            if (xprob[y, 0] == 0):
                cond_cpt[i, 0, 0] = 0
                cond_cpt[i, 1, 0] = 0
            else:
                cond_cpt[i, 0, 0] = xyprob[x, y, 0, 0] / xprob[y, 0]
                cond_cpt[i, 1, 0] = xyprob[x, y, 1, 0] / xprob[y, 0]

            if (xprob[y, 1] == 0):
                cond_cpt[i, 0, 1] = 0
                cond_cpt[i, 1, 1] = 0
            else:
                cond_cpt[i, 0, 1] = xyprob[x, y, 0, 1] / xprob[y, 1]
                cond_cpt[i, 1, 1] = xyprob[x, y, 1, 1] / xprob[y, 1]




            #cond_cpt[i, 0, 0] = xyprob[x, y, 0, 0] / xprob[y, 0]
            #cond_cpt[i, 0, 1] = xyprob[x, y, 0, 1] / xprob[y, 1]
            #cond_cpt[i, 1, 0] = xyprob[x, y, 1, 0] / xprob[y, 0]
            #cond_cpt[i, 1, 1] = xyprob[x, y, 1, 1] / xprob[y, 1]




        if timing:
            print(time.time() - start)
        return cond_cpt

    @staticmethod
    def compute_edge_potential(xyprob, parents, timing=False):

        #print "topo_order: ", topo_order
        #start = time.time()
        nvariables = parents.shape[0]
        edge_potential = np.zeros((nvariables,2,2))

        # for convinient, the first item is redundent
        edge_potential[0, :, :] = 0




        for x in xrange(1, nvariables):
            y = parents[x]

            # id, child, parent

            edge_potential[x, :, :] = xyprob[x, y, :, :]

        return edge_potential

    @staticmethod
    def merge(arr1,arr2,val):
        indices = np.where(arr1==val)[0]
        if indices.shape[0] == arr2.shape[0]:
            np.put(arr1,indices,arr2)

    @staticmethod
    def invperm(p):
        q = np.empty_like(p)
        q[p] = np.arange(len(p))
        return q

    @staticmethod
    def find_map(arr1,arr2):
        o1 = np.argsort(arr1)
        o2 = np.argsort(arr2)
        return o2[Util2.invperm(o1)]

    @staticmethod
    def pos_intersect(arr1,arr2):
        if arr1.size != arr2.size:
            return np.array([])
        diff = np.setdiff1d(arr2,arr1)
        diff_idx = np.where(arr2==diff)[0]
        same_idx = np.setdiff1d(np.arange(arr2.size),diff_idx)
        return np.dstack((arr1[same_idx],arr2[same_idx])).reshape(same_idx.size,2)

    @staticmethod
    def bin_truth_table(nvars):
        unit = np.array([0,1])
        params = np.vstack((2**np.arange(nvars),2**np.arange(nvars)[::-1])).T
        return np.apply_along_axis(lambda arr: np.tile(unit,(arr[0],arr[1])).flatten('F'), 1, params)

    """ Generate indices of table """
    @staticmethod
    def gen_table_idx(index, size, right_block_size, table_size, shift=0):
        start = (index * right_block_size)
        idx = []
        while start <= table_size:
            idx += np.arange(shift + start, shift + start + right_block_size).tolist()
            start += size * right_block_size
        return idx
