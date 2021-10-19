#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 20:28:19 2018

@author: shashajin
"""

import numpy as np
import numba
from Util2 import *
import time

LOG_ZERO = -np.inf

@numba.jit
def get_sample_ll(samples,topo_order, parents, log_cond_cpt):

    nvariables= samples.shape[1]
    probs = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        for j in xrange(nvariables):
            x = topo_order[j]
            assignx=samples[i,x]
            # if root sample from marginal
            if parents[x] == -9999:
                probs[i] += log_cond_cpt[0, assignx, 0]
            else:
                # sample from p(x|y)
                y = parents[x]
                assigny = samples[i,y]
                probs[i] += log_cond_cpt[j, assignx, assigny]
    return probs

@numba.jit
def get_tree_dataset_ll(dataset, topo_order, parents, log_cond_cpt):

    prob = 0.0
    nvariables= dataset.shape[1]
    #print ('compute using log conditional cpt')
    for i in range(dataset.shape[0]):
        for j in xrange(nvariables):
            x = topo_order[j]
            assignx=dataset[i,x]
            # if root sample from marginal
            if parents[x] == -9999:
                prob += log_cond_cpt[0, assignx, 0]
            else:
                # sample from p(x|y)
                y = parents[x]
                assigny = dataset[i,y]
                prob += log_cond_cpt[j, assignx, assigny]
    return prob

@numba.jit
def get_single_ll(sample,topo_order, parents, log_cond_cpt):

    #print sample
    nvariables= sample.shape[0]
    prob = 0.0
    for j in xrange(nvariables):
        x = topo_order[j]
        assignx=sample[x]
        # if root sample from marginal
        if parents[x] == -9999:
            prob += log_cond_cpt[0, assignx, 0]
        else:
            # sample from p(x|y)
            y = parents[x]
            assigny = sample[y]
            prob += log_cond_cpt[j, assignx, assigny]
    return prob




#@numba.jit
def updata_coef(curr_rec, total_rec, lamda, function):

    #print (function)

    ratio = float(curr_rec) / total_rec

    if function == 'linear':
        return lamda * ratio

    if function == 'square':
        return lamda * ratio **(2)

    if function == 'root':
        return lamda * ratio**(0.5)

    #return lamda / (1+np.exp(1-10 * (float(curr_rec) / total_rec)))

    #print ('function not implemented')
    return lamda


@numba.jit
# The varible eliminate for tree structure with only binary variables
def ve_tree_bin_log(topo_order, parents, log_cond_cpt):

    #print (topo_order)
    #print (parents)
    #print (np.exp(log_cond_cpt))
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.zeros((nvariables,2))

    # loop in reverse order, this loop exclude the root
    for i in xrange(nvariables-1, 0, -1):


        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(log_cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.zeros(2) #only handle binary
        cpt_child = cpt_income[x]

        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] += cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.zeros(2)
    root_cpt = cpt_income[0] + log_cond_cpt[0,:,0]

    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    return np.logaddexp(root_cpt[0], root_cpt[1])


@numba.jit
# Using max instead of sum in varible eliminate for tree structure with only binary variables
# return the max probablity as well as
def max_tree_bin_log(topo_order, parents, log_cond_cpt):

    #print (topo_order)
    #print (parents)
    #print (np.exp(log_cond_cpt))
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.zeros((nvariables,2))

    # loop in reverse order, this loop exclude the root
    for i in xrange(nvariables-1, 0, -1):


        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(log_cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.zeros(2) #only handle binary
        cpt_child = cpt_income[x]

        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] += max(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += max(single_cpt[0,1], single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] += cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.zeros(2)
    root_cpt = cpt_income[0] + log_cond_cpt[0,:,0]

    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    return max(root_cpt[0], root_cpt[1])

@numba.jit
# Using max instead of sum in varible eliminate for tree structure with only binary variables
# return the max probablity as well as the map tuple
def max_tree_bin_map(topo_order, parents, log_cond_cpt):

    #print (topo_order)
    #print (parents)
    #print (np.exp(log_cond_cpt))
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.zeros((nvariables,2))
    # This array contains the max assignment of child node given parent value
    # the index in parent assignment, the value is child assignment
    # [1,0] means when p=0, max assginment of  c is 1, when p=1, max assginment of  c is 0
    # based on natual incremental order
    max_reserve_arr = np.zeros((nvariables, 2))

    # loop in reverse order, this loop exclude the root
    for i in xrange(nvariables-1, 0, -1):


        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(log_cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.zeros(2) #only handle binary
        cpt_child = cpt_income[x]

        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        #cpt[0] += max(single_cpt[0,0], single_cpt[1,0])
        #cpt[1] += max(single_cpt[0,1], single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))
        # when tie, always choose 0
        if single_cpt[0,0] >= single_cpt[1,0]:
            max_reserve_arr[x,0] = 0
            cpt[0] += single_cpt[0,0]
        else:
            max_reserve_arr[x,0] = 1
            cpt[0] += single_cpt[1,0]

        if single_cpt[0,1] >= single_cpt[1,1]:
            max_reserve_arr[x,1] = 0
            cpt[1] += single_cpt[0,1]
        else:
            max_reserve_arr[x,1] = 1
            cpt[1] += single_cpt[1,1]



        cpt_income[y] += cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.zeros(2)
    root_cpt = cpt_income[0] + log_cond_cpt[0,:,0]

    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    max_prob = 0.0
    #max_root_asgn = 0
    if root_cpt[0] >= root_cpt[1]:
        max_prob = root_cpt[0]
        max_reserve_arr[0,:] = 0
    else:
        max_prob = root_cpt[1]
        max_reserve_arr[0,:] = 1

    #print ('max_reserve_arr: ', max_reserve_arr)
    # back propgation to find the assignment
    assign_x =  np.zeros(topo_order.shape[0], dtype =int)
    assign_x[0] = max_reserve_arr[0,0]
    for i in xrange(1,topo_order.shape[0]):
        x = topo_order[i]
        y = parents[x]
        #print ('x,y: ', x,y)
        assign_x[x] = max_reserve_arr[x,assign_x[y]]


    #return max(root_cpt[0], root_cpt[1])
    return max_prob, assign_x



@numba.jit
# The varible eliminate for tree structure with only binary variables
def ve_tree_bin(topo_order, parents, cond_cpt):

    #print (topo_order)
    #print (parents)
    #print (np.exp(log_cond_cpt))
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.ones((nvariables,2))

    # loop in reverse order, this loop exclude the root
    for i in xrange(nvariables-1, 0, -1):


        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.ones(2) #only handle binary
        cpt_child = cpt_income[x]

        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] *= cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.zeros(2)
    root_cpt = cpt_income[0] * cond_cpt[0,:,0]


    #print "cpt income" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    return root_cpt[0]+ root_cpt[1]



# The varible eliminate for tree structure with only binary variables
def ve_tree_bin_fast(topo_order, parents, cond_cpt, var1, var2):

    #parents[0] = 0  # temp trick
    #print ('topo_order: ' ,topo_order)
    #print ('parents: ', parents)
    #print ('cond_cpt:')
    #print (cond_cpt)
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.ones((nvariables,2))
    cpt_income_extra = np.ones((2,2,2)) # store the cpt that contain var1, var2 as child
    # the current parenents for var1 and var2
    curr_p_var = np.array([parents[var1], parents[var2]])
    ind = np.zeros(2, dtype = int)
    ind[0] =  np.argwhere (topo_order == var1)# the index in topo for var1
    ind[1] =  np.argwhere (topo_order == var2)# the index in topo for var2
    #print ('ind: ', ind)


    cpt_income_extra[0] = cond_cpt[ind[0]]
    cpt_income_extra[1] = cond_cpt[ind[1]]
    #if parents[var2] == var1


    ind_remain = np.delete(np.arange(nvariables), ind)
    elimi_order = topo_order[ind_remain]
    elimi_cpt = cond_cpt[ind_remain]

    #print ('curr_p_var: ', curr_p_var)
    #print ('cpt_income_extra: ')
    #print (cpt_income_extra)
    #print ('elimi_order: ', elimi_order)
    #print ('elimi_cpt')
    #print (elimi_cpt)


    if elimi_order[0] == 0:
        last_ind = 0
    else:
        last_ind = -1

    # loop in reverse order, this loop exclude the root
    for i in xrange(elimi_order.shape[0]-1, last_ind, -1):


        x = elimi_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(elimi_cpt[i])
        #print "single cpt: ", single_cpt
        cpt = np.ones(2) #only handle binary
        cpt_child = cpt_income[x]

        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]

        # x has child that is var1 or var2, here x may not be the direct parents, could be ancestor
        if x in curr_p_var:
            # has the same ancenstor
            if curr_p_var.shape[0] == 1:
                cpt_income_extra = np.einsum('lk,ijl ->ijk',single_cpt,cpt_income_extra)
                curr_p_var[0] = y


            else:

                # the cloest ancestor
                if curr_p_var[0] == curr_p_var[1]:
                    cpt_income_extra = np.einsum('lk,il, jl ->ijk',single_cpt,cpt_income_extra[0], cpt_income_extra[1])
                    curr_p_var = np.array([y])

                else:
                    if x == curr_p_var[0]:
                        cpt_income_extra[0] = np.einsum('ki,jk ->ji',single_cpt,cpt_income_extra[0])
                        curr_p_var[0] = y

                    elif x == curr_p_var[1]:
                         cpt_income_extra[1] = np.einsum('ki,jk ->ji',single_cpt,cpt_income_extra[1])
                         curr_p_var[1] = y



        else:
            #print "cpt before: ", np.exp( cpt)
            cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
            cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])

            #print ("cpt: ", np.exp(cpt))

            cpt_income[y] *= cpt

        #print "cpt_income"
        #print  cpt_income


    #print 'cpt_income_extra:'
    #print cpt_income_extra
    # the root node
    #print ('curr_p_var: ', curr_p_var)
    if var1 == 0:
        pxy = np.einsum('i,i, ji ->ij',cpt_income[0],cpt_income_extra[0,:,0], cpt_income_extra[1])
        #print ('1')
        #print pxy
        pxy = np.einsum('j, ij ->ij',cpt_income[var2],pxy)
        #print ('2')
        #print pxy

    elif var2==0:
        pxy = np.einsum('j,j, ij ->ij',cpt_income[0],cpt_income_extra[1,:,0], cpt_income_extra[0])
        pxy = np.einsum('i, ij ->ij',cpt_income[var1],pxy)
    # sumout the root value
    else:
        # root is not the first ancestor of var1 and var2
        # we have var1, var2, 0 in cpt_income_extra
        if curr_p_var.shape[0] == 1:
            pxy = np.einsum('k,k, ijk ->ij',cpt_income[0],cond_cpt[0,:,0], cpt_income_extra)
        else:
            if curr_p_var[0] == 0:
                if curr_p_var[1] == 0:  # we have var1, 0 and var2, 0
                    pxy = np.einsum('k,k, ik, jk ->ij',cpt_income[0],cond_cpt[0,:,0], cpt_income_extra[0], cpt_income_extra[1])
                else:               # we have va1, 0 and var2, var1
                    pxy = np.einsum('k,k, ik, ji ->ij',cpt_income[0],cond_cpt[0,:,0], cpt_income_extra[0], cpt_income_extra[1])
            else:         # we have va1, var2 and var2, 0
                pxy = np.einsum('k,k, ij, jk ->ij',cpt_income[0],cond_cpt[0,:,0], cpt_income_extra[0], cpt_income_extra[1])
            #print ('pxy')
            #print pxy
        pxy = np.einsum('i,j, ij ->ij',cpt_income[var1],cpt_income[var2], pxy)



    return pxy



@numba.jit
# The varible eliminate for tree structure with only binary variables
# compute P(0,0), P(0,1), P(1,0), P(1,1) at the same time
def ve_tree_bin2(topo_order, parents, cond_cpt, var1, var2, cpt_income_orig):

    #print (topo_order)
    #print (parents)
    #print (np.exp(log_cond_cpt))
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.copy(cpt_income_orig)
    cpt_income_save =  np.ones((3,cpt_income.shape[0],2))
    topo_loc = np.zeros(2, dtype=np.uint32)
    p_xy = np.zeros((2,2))
    flag = False
    #print 'cpt_income:', cpt_income
    #-------------------------------------------------------
    # (0, 0)  along the topo_order
    #-------------------------------------------------------
    # loop in reverse order, this loop exclude the root
    for i in xrange(nvariables-1, 0, -1):


        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.ones(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])

        if x==var1 or x== var2:
            # locate the x in topo_order
            if flag == False:
                topo_loc[0] = i
                cpt_income_save[1] = np.copy(cpt_income) # for 11
                flag = True
            else:
                topo_loc[1] = i
                cpt_income_save[0] = np.copy(cpt_income) # for 01
            #set x = 0
            # x as child
            single_cpt[1,:] = 0
            # x as parent
            cpt_child[1] = 0


        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] *= cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    #print root_cpt
    root_cpt_income = np.copy(cpt_income[0])
    #print root_cpt_income

    if topo_order[0] == var1 or topo_order[0] == var2:
        cpt_income_save[0] = np.copy(cpt_income) # for 01, special case
        root_cpt[1] = 0
        root_cpt_income[1] = 0

    root_cpt *= root_cpt_income


    #print "cpt income after 0,0 :" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    p_xy[0,0] =  root_cpt[0]+ root_cpt[1]

    #print ("topo_loc: ", topo_loc)

    #
    #-------------------------------------------------------
    # (1, 0) along the topo_order
    #-------------------------------------------------------
    #for i in xrange(nvariables-1, 0, -1):
    #print "before:"
    #print cpt_income
    cpt_income = cpt_income_save[0]
    #print "after:"
    #print cpt_income
    for i in xrange(topo_loc[1], 0, -1):

        #print "x: ", x
        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.ones(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])

        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[0,:] = 0
            # x as parent
            cpt_child[0] = 0


        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] *= cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    #print root_cpt_income

    if topo_order[0] == var1 or topo_order[0] == var2:
        root_cpt[0] = 0
        root_cpt_income[0] = 0

    root_cpt *= root_cpt_income

    #root_cpt = cpt_income[0] * cond_cpt[0,:,0]


    #print "cpt income after 0,1 :" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    if topo_order[topo_loc[1]] == var1:
        p_xy[1,0] =  root_cpt[0]+ root_cpt[1]
    else:
        p_xy[0,1] =  root_cpt[0]+ root_cpt[1]


    #-------------------------------------------------------
    # (1,1) along the topo_order
    #-------------------------------------------------------
    #cpt_income = np.ones((nvariables,2))
    #for i in xrange(nvariables-1, 0, -1):
    cpt_income = cpt_income_save[1]
    for i in xrange(topo_loc[0], 0, -1):


        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.ones(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])

        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[0,:] = 0
            # x as parent
            cpt_child[0] = 0

            if i == topo_loc[1]:
                cpt_income_save[2] = np.copy(cpt_income) # for 01



        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] *= cpt

        #print np.exp(cpt_income)


    #print "cpt income after 1,1 :" , cpt_income
   # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    #print root_cpt
    root_cpt_income = np.copy(cpt_income[0])
    #print root_cpt_income

    if topo_order[0] == var1 or topo_order[0] == var2:
        cpt_income_save[2] = np.copy(cpt_income) # for 10, special case
        root_cpt[0] = 0
        root_cpt_income[0] = 0

    root_cpt *= root_cpt_income


    #print "cpt income:" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    p_xy[1,1] =  root_cpt[0]+ root_cpt[1]


    #-------------------------------------------------------
    # (0,1) along the topo_order
    #-------------------------------------------------------
    cpt_income = cpt_income_save[2]
    for i in xrange(topo_loc[1], 0, -1):

        #print "x: ", x
        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.ones(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])

        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[1,:] = 0
            # x as parent
            cpt_child[1] = 0


        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] *= cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    #print root_cpt_income

    if topo_order[0] == var1 or topo_order[0] == var2:
        root_cpt[1] = 0
        root_cpt_income[1] = 0

    root_cpt *= root_cpt_income

    #root_cpt = cpt_income[0] * cond_cpt[0,:,0]


    #print "cpt income after 0,1 :" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    if topo_order[topo_loc[1]] == var1:
        p_xy[0,1] =  root_cpt[0]+ root_cpt[1]
    else:
        p_xy[1,0] =  root_cpt[0]+ root_cpt[1]


    #print p_xy

    #p_xy = p_xy / np.max(p_xy)
    #p_xy += 1e-8

    #return p_xy / np.sum(p_xy)  # normalize

    # Do Not normalize
    return p_xy




@numba.jit
def ve_tree_bin_log2(topo_order, parents, log_cond_cpt, var1, var2):

    #print (topo_order)
    #print (parents)
    #print (np.exp(log_cond_cpt))
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.zeros((nvariables,2))
    cpt_income_save =  np.zeros((3,nvariables,2))
    topo_loc = np.zeros(2, dtype=np.uint32)
    p_xy = np.zeros((2,2))
    flag = False

    #-------------------------------------------------------
    # (0, 0)  along the topo_order
    #-------------------------------------------------------
    # loop in reverse order, this loop exclude the root
    for i in xrange(nvariables-1, 0, -1):


        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(log_cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.zeros(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])

        if x==var1 or x== var2:
            # locate the x in topo_order
            if flag == False:
                topo_loc[0] = i
                cpt_income_save[1] = np.copy(cpt_income) # for 11
                flag = True
            else:
                topo_loc[1] = i
                cpt_income_save[0] = np.copy(cpt_income) # for 01
            #set x = 0
            # x as child
            single_cpt[1,:] = LOG_ZERO
            # x as parent
            cpt_child[1] = LOG_ZERO


        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] += cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.copy(log_cond_cpt[0,:,0])
    #print root_cpt
    root_cpt_income = np.copy(cpt_income[0])
    #print root_cpt_income

    if topo_order[0] == var1 or topo_order[0] == var2:
        cpt_income_save[0] = np.copy(cpt_income) # for 01, special case
        root_cpt[1] = LOG_ZERO
        root_cpt_income[1] = LOG_ZERO

    root_cpt += root_cpt_income


    #print "cpt income after 0,0 :" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    p_xy[0,0] =  np.logaddexp(root_cpt[0], root_cpt[1])

    #print ("topo_loc: ", topo_loc)

    #
    #-------------------------------------------------------
    # (1, 0) along the topo_order
    #-------------------------------------------------------
    #for i in xrange(nvariables-1, 0, -1):
    #print "before:"
    #print cpt_income
    cpt_income = cpt_income_save[0]
    #print "after:"
    #print cpt_income
    for i in xrange(topo_loc[1], 0, -1):

        #print "x: ", x
        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(log_cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.zeros(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])

        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[0,:] = LOG_ZERO
            # x as parent
            cpt_child[0] = LOG_ZERO


        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] += cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.copy(log_cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    #print root_cpt_income

    if topo_order[0] == var1 or topo_order[0] == var2:
        root_cpt[0] = LOG_ZERO
        root_cpt_income[0] = LOG_ZERO

    root_cpt += root_cpt_income

    #root_cpt = cpt_income[0] * cond_cpt[0,:,0]


    #print "cpt income after 0,1 :" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    if topo_order[topo_loc[1]] == var1:
        p_xy[1,0] =  np.logaddexp(root_cpt[0], root_cpt[1])
    else:
        p_xy[0,1] =  np.logaddexp(root_cpt[0], root_cpt[1])


    #-------------------------------------------------------
    # (1,1) along the topo_order
    #-------------------------------------------------------
    #cpt_income = np.ones((nvariables,2))
    #for i in xrange(nvariables-1, 0, -1):
    cpt_income = cpt_income_save[1]
    for i in xrange(topo_loc[0], 0, -1):


        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(log_cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.zeros(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])

        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[0,:] = LOG_ZERO
            # x as parent
            cpt_child[0] = LOG_ZERO

            if i == topo_loc[1]:
                cpt_income_save[2] = np.copy(cpt_income) # for 01



        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] += cpt

        #print np.exp(cpt_income)


    #print "cpt income after 1,1 :" , cpt_income
   # the root node:
    root_cpt = np.copy(log_cond_cpt[0,:,0])
    #print root_cpt
    root_cpt_income = np.copy(cpt_income[0])
    #print root_cpt_income

    if topo_order[0] == var1 or topo_order[0] == var2:
        cpt_income_save[2] = np.copy(cpt_income) # for 10, special case
        root_cpt[0] = LOG_ZERO
        root_cpt_income[0] = LOG_ZERO

    root_cpt += root_cpt_income


    #print "cpt income:" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    p_xy[1,1] =  np.logaddexp(root_cpt[0], root_cpt[1])


    #-------------------------------------------------------
    # (0,1) along the topo_order
    #-------------------------------------------------------
    cpt_income = cpt_income_save[2]
    for i in xrange(topo_loc[1], 0, -1):

        #print "x: ", x
        x = topo_order[i]

        #print "process node: ", x
        y = parents[x]
        #print "parent node: ", y
        single_cpt = np.copy(log_cond_cpt[i])
        #print "single cpt: ", np.exp(single_cpt)
        cpt = np.zeros(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])

        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[1,:] = LOG_ZERO
            # x as parent
            cpt_child[1] = LOG_ZERO


        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]

        #print "cpt before: ", np.exp( cpt)
        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])

        #print ("cpt: ", np.exp(cpt))

        cpt_income[y] += cpt

        #print np.exp(cpt_income)



    # the root node:
    root_cpt = np.copy(log_cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    #print root_cpt_income

    if topo_order[0] == var1 or topo_order[0] == var2:
        root_cpt[1] = LOG_ZERO
        root_cpt_income[1] = LOG_ZERO

    root_cpt += root_cpt_income

    #root_cpt = cpt_income[0] * cond_cpt[0,:,0]


    #print "cpt income after 0,1 :" , cpt_income
    #print "root_cpt", root_cpt
    #print np.exp(np.logaddexp(root_cpt[0], root_cpt[1]))
    if topo_order[topo_loc[1]] == var1:
        p_xy[0,1] =  np.logaddexp(root_cpt[0], root_cpt[1])
    else:
        p_xy[1,0] =  np.logaddexp(root_cpt[0], root_cpt[1])


    #print p_xy
    #p_xy -= np.max(p_xy)
    #return Util.normalize(np.exp(p_xy))  # normalize
    return p_xy

#@numba.jit
def get_prob_matrix(topo_order, parents, cond_cpt, ids, tree_path):
    #print topo_order
    #print parents
    #dim = ids[-1]+1
    #print tree_path
    dim = topo_order.shape[0]
    p_xy = np.zeros((dim, dim, 2, 2))

    """
    min_value = np.min(cond_cpt)
    if (min_value < 0):
        print "Error...Probability can not be less than 0)"
        exit()
    """

    #print 'get_prob_matrix'

    nvariables= topo_order.shape[0]
    cpt_income = np.ones((nvariables,2))

    #print   tree_path

    evid_path_list = []
    # has evidence
    if ids.shape[0] < nvariables:
        evidence = np.setdiff1d(np.arange(nvariables), ids)
        evidence_pos = np.where(np.isin(topo_order, evidence))[0]
        for pos in evidence_pos:
            #print pos
            evid = topo_order[pos]
            #print pos, evid



            if cond_cpt[pos,0,0] + cond_cpt[pos,0,1]  == 0:
                cpt_income[evid, 0] = 0
            else:
                cpt_income[evid, 1] = 0

            evid_path_list += tree_path[evid]

    #print 'cpt_income:', cpt_income
    evid_set = set(evid_path_list)
    #evid_arr = np.unique(np.asarray(evid_path_list))
    #p_0_2 = ve_tree_bin2(topo_order, parents, cond_cpt, 0, 2)
    #print p_0_2
    #p_12_15 = ve_tree_bin2(topo_order, parents, cond_cpt, 12, 15)
    #print p_12_15
    #p_12_8 = ve_tree_bin2(topo_order, parents, cond_cpt, 12, 8)
    #print p_12_8
    #print ('evid_set: ', evid_set)
    for i, x in enumerate(ids):
        # find edges that x is the parent
        x_set = set(tree_path[x]).union(evid_set)
        #print np.asarray(tree_path[x])
        #x_arr = np.append(evid_arr, np.asarray(tree_path[x]))
        for j in xrange(i+1, ids.shape[0]):

            y = ids[j]



            #print ('x,y', x, y)
            #print x, y
            y_set = set(tree_path[y]).union(x_set)
            #y_arr = np.unique(np.append(x_arr, np.asarray(tree_path[y])))
            #print y_set
            ind = np.where(np.isin(topo_order, np.array(list(y_set))))
            #ind = list(topo_order).index(list(y_set))
            #ind = np.where(np.isin(topo_order, y_arr))
            #print ('ind: ', ind)
            new_topo_order = topo_order[ind]
            #print new_topo_order
            new_cond_cpt = cond_cpt[ind]
            #print new_cond_cpt
            #print '1', time.time() - start
            p_xy[x,y,:,:] = ve_tree_bin2(new_topo_order, parents, new_cond_cpt, x, y, cpt_income)
            #print '2', time.time() -start

            #p_xy[x,y,:,:] = ve_tree_bin_fast(topo_order, parents, cond_cpt, x, y) # no benefit, only works when dataset has small variables
            p_xy[y,x,:,:] = p_xy[x,y,:,:]
            # swap
            p_xy[y,x,0,1], p_xy[y,x,1,0] = p_xy[y,x,1,0], p_xy[y,x,0,1]

    p_xy = p_xy[:,ids][ids,:]
    #print p_xy.shape
    #print p_xy[15,12]
    #print p_xy[12,15]
    #print p_xy[0,2]
    #print p_xy[2,0]
    #print ("pxy7:", p_xy[7])

    """
    # compute p_x
    p_x = np.zeros((ids.shape[0], 2))
    p_x[:,0] = p_xy[0,:,0,0] + p_xy[0,:,1,0]
    p_x[:,1] = p_xy[0,:,0,1] + p_xy[0,:,1,1]

    p_x[0,0] = p_xy[1,0,0,0] + p_xy[1,0,1,0]
    p_x[0,1] = p_xy[1,0,0,1] + p_xy[1,0,1,1]

    #print ("p_x")
    #print p_x


    for i in xrange (ids.shape[0]):
        p_xy[i,i,0,0] = p_x[i,0] - 1e-8
        p_xy[i,i,1,1] = p_x[i,1] - 1e-8
        p_xy[i,i,0,1] = 1e-8
        p_xy[i,i,1,0] = 1e-8


    #for i in xrange (ids.shape[0]):
    #    print p_xy[i,i,:,:]

    return p_xy, p_x
    """

    return p_xy

def get_prob_matrix_log(topo_order, parents, log_cond_cpt, ids):
    #print topo_order
    #print parents
    #dim = ids[-1]+1
    dim = topo_order.shape[0]
    p_xy = np.zeros((dim, dim, 2, 2))

    """
    min_value = np.min(cond_cpt)
    if (min_value < 0):
        print "Error...Probability can not be less than 0)"
        exit()
    """




    #p_0_2 = ve_tree_bin2(topo_order, parents, cond_cpt, 0, 2)
    #print p_0_2
    #p_12_15 = ve_tree_bin2(topo_order, parents, cond_cpt, 12, 15)
    #print p_12_15
    #p_12_8 = ve_tree_bin2(topo_order, parents, cond_cpt, 12, 8)
    #print p_12_8

    for i, x in enumerate(ids):
        # find edges that x is the parent
        for j in xrange(i+1, ids.shape[0]):
            y = ids[j]
            #print x, y
            p_xy[x,y,:,:] = ve_tree_bin_log2(topo_order, parents, log_cond_cpt, x, y)
            p_xy[y,x,:,:] = p_xy[x,y,:,:]
            # swap
            p_xy[y,x,0,1], p_xy[y,x,1,0] = p_xy[y,x,1,0], p_xy[y,x,0,1]

    p_xy = p_xy[:,ids][ids,:]
    #print p_xy.shape
    #print p_xy[15,12]
    #print p_xy[12,15]
    #print p_xy[0,2]
    #print p_xy[2,0]
    #print ("pxy7:", p_xy[7])


    # compute p_x
    p_x = np.zeros((ids.shape[0], 2))
    p_x[:,0] = p_xy[0,:,0,0] + p_xy[0,:,1,0]
    p_x[:,1] = p_xy[0,:,0,1] + p_xy[0,:,1,1]

    p_x[0,0] = p_xy[1,0,0,0] + p_xy[1,0,1,0]
    p_x[0,1] = p_xy[1,0,0,1] + p_xy[1,0,1,1]

    #print ("p_x")
    #print p_x

    for i in xrange (ids.shape[0]):
        p_xy[i,i,0,0] = p_x[i,0] - 1e-8
        p_xy[i,i,1,1] = p_x[i,1] - 1e-8
        p_xy[i,i,0,1] = 1e-8
        p_xy[i,i,1,0] = 1e-8

    #for i in xrange (ids.shape[0]):
    #    print p_xy[i,i,:,:]

    return p_xy, p_x

def save_cutset(main_dict, node, ids, ccpt_flag = False):
    if isinstance(node,list):
        id,x,p0,p1,node0,node1=node
        main_dict['type'] = 'internal'
        main_dict['id'] = id
        main_dict['x'] = x
        main_dict['p0'] = p0
        main_dict['p1'] = p1
        main_dict['c0'] = {}  # the child associated with p0
        main_dict['c1'] = {}  # the child associated with p0

        ids=np.delete(ids,id,0)
        save_cutset(main_dict['c0'], node0, ids, ccpt_flag)
        save_cutset(main_dict['c1'], node1, ids, ccpt_flag)
    else:
        main_dict['type'] = 'leaf'

        if ccpt_flag == False:
            node.get_log_cond_cpt()
        main_dict['log_cond_cpt'] =  node.log_cond_cpt
        main_dict['topo_order'] = node.topo_order
        main_dict['parents'] = node.parents
        #main_dict['ids'] = node.ids           #2
        #main_dict['p_xy'] = node.xyprob          #3
        main_dict['p_x'] = node.xprob           #4
        return

def computeLL_reload(reload_cutset, dataset):
    probs = np.zeros(dataset.shape[0])
    #cnet = copy.deepcopy(reload_cutset)
    for i in range(dataset.shape[0]):
        cnet = reload_cutset
        prob = 0.0
        #node=self.tree
        ids=np.arange(dataset.shape[1])
        #print (cnet['type'])
        while cnet['type'] == 'internal':
            id = cnet['id']
            x  = cnet['x']
            p0 = cnet['p0']
            p1 = cnet['p1']
            c0 = cnet['c0']
            c1 = cnet['c1']

            assignx=dataset[i,x]
            ids=np.delete(ids,id,0)
            if assignx==1:
                prob+=np.log(p1/(p0+p1))
                cnet=c1
            else:
                prob+=np.log(p0/(p0+p1))
                cnet=c0
            #print ('x:',x)
        # reach the leaf clt
        if cnet['type'] == 'leaf':
            #clt = CLT()
            log_cond_cpt = cnet['log_cond_cpt']
            topo_order = cnet['topo_order']
            parents = cnet['parents']
            #clt.xyprob = cnet['p_xy']
            #clt.xprob = cnet['p_x']
            #get_sample_ll(samples,topo_order, parents, log_cond_cpt)
            #print ('leaf prob: ', get_single_ll(dataset[i][ids], topo_order, parents, log_cond_cpt))
            #print ('leaf ids: ' , ids)
            #print (log_cond_cpt)
            prob += get_single_ll(dataset[i][ids], topo_order, parents, log_cond_cpt)
            probs[i] = prob
            #print ('b:', prob)
        else:
            print ("*****ERROR******")
            exit()

    return probs


@numba.jit
#-------------------------------------------------------------------------------
# log space subtraction
# return log (exp(x) - exp (y)) if x > y
#-------------------------------------------------------------------------------
def log_subtract(x, y):
    if(x < y):
        print ("Error!! computing the log of a negative number \n")
        #return np.log(-1)
        # under our assumption, x < y could not happen, if happens, we believe it is caused by numeric issue
        return LOG_ZERO
    if (x == y) :
        return LOG_ZERO

    return x + np.log1p(-np.exp(y-x))


@numba.jit
#-------------------------------------------------------------------------------
# Add an array in log space
#-------------------------------------------------------------------------------
def log_add_arr(log_arr):
    sum_val = LOG_ZERO
    for i in xrange(log_arr.shape[0]):
        sum_val = np.logaddexp(sum_val, log_arr[i])

    return sum_val



#@numba.jit
#def bin_to_dec (bin_arr):
#    return np.sum(2**np.arange(bin_arr.shape[0] - 1, -1, -1)*bin_arr)

@numba.jit
def get_labels (input_arr):
    n_record = input_arr.shape[0]
    label_arr = np.zeros(n_record, dtype= int)
    for i in xrange(n_record):
        label_arr[i]= 2*input_arr[i,0] +input_arr[i,1]
    return label_arr


"""
topo_order = np.arange(4)
parents = np.arange(4)-1
#print topo_order
#print parents
"""

"""
topo_order = np.array([0,2,1,3])
parents = np.array([-9999,0,0,1])

ids = np.arange(4)

cond_cpt = np.zeros((4,2,2))
cond_cpt[0,0,0] = 0.3
cond_cpt[0,0,1] = 0.3
cond_cpt[0,1,0] = 0.7
cond_cpt[0,1,1] = 0.7
cond_cpt[1,0,0] = 0.2
cond_cpt[1,0,1] = 0.4
cond_cpt[1,1,0] = 0.8
cond_cpt[1,1,1] = 0.6
cond_cpt[2,0,0] = 0.3
cond_cpt[2,0,1] = 0.1
cond_cpt[2,1,0] = 0.7
cond_cpt[2,1,1] = 0.9
cond_cpt[3,0,0] = 0.8
cond_cpt[3,0,1] = 0.7
cond_cpt[3,1,0] = 0.2
cond_cpt[3,1,1] = 0.3
"""

#var1 = 0
#var2 = 2
#print "slow version: ", ve_tree_bin2(topo_order, parents, cond_cpt, var1 , var2)
#p = ve_tree_bin_fast(topo_order, parents, cond_cpt, var1 , var2)
#print "fast version: ", p

#print (ve_tree_bin2(topo_order, parents, cond_cpt, 0 , 3))
#print (ve_tree_bin_log2(topo_order, parents, np.log(cond_cpt), 0 , 3))

"""
import time
start = time.time()
p = get_prob_matrix(topo_order, parents, cond_cpt, ids)
print p
print ('running time: ', time.time()-start)
"""

"""
variable_id = 1
index_c = np.where(topo_order==variable_id)[0][0]
# variable as parent
varible_child = np.where(parents ==variable_id)[0]
ix = np.isin(topo_order, varible_child)
index_p = np.where(ix)[0]
#print (index_p)
"""

"""
 # set varible value = 1
cond_cpt_1 = np.copy(cond_cpt)
cond_cpt_1[index_c, 0,:] = 0
cond_cpt_1[index_p, :,0] = 0

#print cond_cpt_1
print(ve_tree_bin2(topo_order, parents, cond_cpt_1, 0 , 3))
print (ve_tree_bin_fast(topo_order, parents, cond_cpt_1, 0,3))
"""

"""
 # set varible value = 1
log_cond_cpt_1 = np.copy(np.log(cond_cpt))
log_cond_cpt_1[index_c, 0,:] = LOG_ZERO
log_cond_cpt_1[index_p, :,0] = LOG_ZERO

#print cond_cpt_1
#print(ve_tree_bin_log2(topo_order, parents, cond_cpt_1, 0 , 3))
log_cond_cpt_0 = np.copy(np.log(cond_cpt))
log_cond_cpt_0[index_c, 1,:] = LOG_ZERO
log_cond_cpt_0[index_p, :,1] = LOG_ZERO

log_cond_cpt = np.log(cond_cpt)

#print 've log:',ve_tree_bin_log(topo_order, parents, np.log(cond_cpt))
#print 've:',ve_tree_bin(topo_order, parents, cond_cpt)
print 'max log:',max_tree_bin_log(topo_order, parents, log_cond_cpt_0)
print 'max map:',max_tree_bin_map(topo_order, parents, log_cond_cpt_0)


log_cond_cpt = np.copy(np.log(cond_cpt))
samples = np.zeros((16,4), dtype = int)
samples[0] = np.array([0,0,0,0])
samples[1] = np.array([0,0,0,1])
samples[2] = np.array([0,0,1,0])
samples[3] = np.array([0,0,1,1])
samples[4] = np.array([0,1,0,0])
samples[5] = np.array([0,1,0,1])
samples[6] = np.array([0,1,1,0])
samples[7] = np.array([0,1,1,1])
samples[8] = np.array([1,0,0,0])
samples[9] = np.array([1,0,0,1])
samples[10] = np.array([1,0,1,0])
samples[11] = np.array([1,0,1,1])
samples[12] = np.array([1,1,0,0])
samples[13] = np.array([1,1,0,1])
samples[14] = np.array([1,1,1,0])
samples[15] = np.array([1,1,1,1])

print samples
print get_sample_ll(samples,topo_order, parents, log_cond_cpt)
"""
