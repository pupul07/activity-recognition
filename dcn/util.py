from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dcn.Util2

# Remove weak edges from a CLT
def remove_weak_edges(mi_matrix, topo_order, parents, thres=0.05):
  for t in topo_order:
    if parents[t]<0:
      continue
    parents[t] = parents[t] if mi_matrix[t,parents[t]]>=thres else -9999

# Generate mutual information matrix
def gen_mi_matrix(dset, continuous_flag=False):
  if(len(dset.shape)<2):
    return np.array([[]])
  mi_matrix = np.zeros((dset.shape[1],dset.shape[1]))
  if (dset.shape[0]==0):
      return mi_matrix
  for i in range(mi_matrix.shape[0]):
    for j in range(i+1,mi_matrix.shape[1]):
      if(continuous_flag):
        mi_matrix[i,j] = mutual_info_regression(dset[:,i:i+1],dset[:,j], n_neighbors=5, random_state=1991)
      else:
        mi_matrix[i,j] = mutual_info_score(dset[:,i],dset[:,j])
      mi_matrix[j,i] = mi_matrix[i,j]
  return mi_matrix

# Weighted version of gen_mi_matrix
def gen_weighted_mi_matrix(dset, weights):
    if (dset.ndim != 2 or weights.ndim != 1 or dset.shape[0]!=weights.size or weights.sum()==0):
        return np.array([[]])
    xycounts = Util2.Util2.compute_weighted_xycounts(dset, weights) + Util2.SMOOTH
    xcounts = Util2.Util2.compute_weighted_xcounts(dset, weights) + (2*Util2.SMOOTH)
    mi_matrix = Util2.Util2.compute_edge_weights(xycounts, xcounts)
    return mi_matrix

# Generate mutual information matrix
def gen_mi_matrix_select(dset, select_idx=np.array([]), continuous_flag=False):
  if(len(dset.shape)<2 or select_idx.ndim!=1):
    return np.array([[]])
  if(select_idx.size==0):
    return gen_mi_matrix(dset, continuous_flag)
  mi_matrix = np.zeros((dset.shape[1],select_idx.size))
  if (dset.shape[0]==0):
      return mi_matrix
  for i in np.arange(dset.shape[1]):
    for j in np.arange(select_idx.size):
      s = select_idx[j]
      if(continuous_flag):
        mi_matrix[i,j] = mutual_info_regression(dset[:,i:i+1],dset[:,s], n_neighbors=5, random_state=1991)
      else:
        mi_matrix[i,j] = mutual_info_score(dset[:,i],dset[:,s])
  return mi_matrix

# Weighted version of gen_mi_matrix_select
def gen_weighted_mi_matrix_select(dset, weights, select_idx=np.array([])):
  if(dset.ndim != 2 or weights.ndim != 1 or dset.shape[0] != weights.size or \
          weights.sum() == 0 or select_idx.ndim!=1):
    return np.array([[]])
  if(select_idx.size==0):
    return gen_weighted_mi_matrix(dset, weights)
  if (dset.shape[0]==0):
      return np.zeros((dset.shape[1],select_idx.size))
  xycounts = Util2.Util2.compute_xycounts_new(dset, select_idx, weights)[:,select_idx] + Util2.SMOOTH
  xcounts = Util2.Util2.compute_xcounts_new(dset, weights=weights) + (2 * Util2.SMOOTH)
  ycounts = Util2.Util2.compute_xcounts_new(dset[:,select_idx], weights=weights) + (2 * Util2.SMOOTH)
  mi_matrix = Util2.Util2.compute_edge_weights_new(xycounts, xcounts, ycounts)
  return mi_matrix

# Given a set of values and the domains, generate address
def calc_address(vals, doms=np.array([])):
  if (vals.ndim != 1):
    return -1
  addrs = calc_addresses(vals[np.newaxis,:], doms)
  addr = addrs[0] if addrs.size>0 else -1
  return addr

# Given a set of values and the domains, generate address vector
def calc_addresses(vals, doms=np.array([])):
  if not doms.size:
    doms = np.full(vals.shape[0], 2, dtype=int)
  if (vals.ndim != 2 or doms.ndim !=1 or vals.shape[1] != doms.size):
    return np.array([])
  addrs = np.zeros(vals.shape[0], dtype=int)
  cum_doms = np.append(np.cumprod(doms[::-1])[::-1][1:],1)
  for v in range(doms.size):
    addrs += cum_doms[v] * vals[:,v]
  return addrs

# Given an address and the domains, calculate the value assignments
def calc_vals_from_address(addr, doms):
  # Validation
  if doms.ndim != 1 or addr<0 or addr>=doms.prod():
    return np.array([], dtype=int)
  # Initialization
  vals = np.zeros(doms.size, dtype=int)
  block_sizes = np.append(doms[::-1].cumprod()[::-1][1:],1)
  # Calculate vals
  for v in range(vals.size):
    vals[v] = int(addr/block_sizes[v])
    addr = addr%block_sizes[v]
    if (addr==0):
      break
  # Return vals
  return vals

# Given an assignment, goes to next assignment
def go_to_next_assign(curr_assign, doms):
  # If invalid, do nothing
  if(doms.shape[0] != curr_assign.shape[0] or curr_assign.shape[0]==0):
    return
  # Find index to fill from
  start = np.append(-1,np.argwhere(1-(curr_assign+1==doms)))[-1]
  # Fill everything after start with 0
  curr_assign[start+1:].fill(0)
  # Increment element at start index by 1 as long as it's not -1
  curr_assign[start] += 1 if(start>=0) else 0

# Same as go_to_next_assign but returns a new ndarray
def calc_next_assign(curr_assign, doms):
    new_assign = np.array(curr_assign)
    go_to_next_assign(new_assign, doms)
    return new_assign

# Function to draw a DBN model and save it as a .png file
def draw_dbn_model(dir_path, file_name, doms, scopes):
    # Initialize graph
    G = nx.DiGraph()
    # Add all nodes
    V = doms.shape[0]
    node_labels = {}
    for v in range(2*V):
        suffix = "_t-1" if v<V else "_t"
        node_labels[v] = "x" + str(v%V) + suffix
        G.add_node(v)
    # Add edges from scope
    for scope in scopes:
        i = scope[-1]
        for j in scope[:-1]:
            G.add_edge(j,i)
    # Save model
    outfile_path = dir_path + "/" + file_name + ".png"
    nx.draw(G, with_labels=True, labels=node_labels, font_size=8,
            node_size=1000, node_color=['b']*V+['r']*V)
    plt.savefig(outfile_path)
    plt.clf()

# Function to load DBN model from file
def load_dbn_model(dir_path, file_name):
    # Initialize variables
    infile_path = dir_path + "/" + file_name + ".dbn"
    doms = np.array([], dtype=int)
    scopes = []
    cpts = []
    with open(infile_path, "r") as infile:
        lines = infile.readlines()
        V = int(lines[0]) if len(lines) else 1
        if (len(lines) >= (4*V)+2):
            # Read in doms
            doms = np.fromstring(lines[1].strip(), dtype=int, sep=",")
            # Read in scopes
            for line in lines[2:(2*V)+2]:
                scopes += [ np.fromstring(line.strip(), dtype=int, sep=",") ]
            # Read in CPTs
            for line in lines[(2*V)+2:(4*V)+2]:
                cpts += [ np.fromstring(line.strip(), dtype=float, sep=",") ]
    return (doms, scopes, cpts)

# Function to save DBN model to file
def save_dbn_model(dir_path, file_name, doms, scopes, cpts):
    # Initialize variables
    outfile_path = dir_path + "/" + file_name + ".dbn"
    with open(outfile_path, "w") as outfile:
        # Write number of variables
        outfile.write(str(doms.size)+"\n")
        # Write domains
        outfile.write(', '.join(map(str,doms))+"\n")
        # Write scopes
        for scope in scopes:
            outfile.write(', '.join(map(str,scope))+"\n")
        # Write CPTs
        for cpt in cpts:
            outfile.write(', '.join(map(str,cpt))+"\n")

# Computes the probability of evidence of a DBN where everything is sorted topologically
def dbn_prob_evid(doms, scopes, cpts, evid_idx, evid_vals):
    if (evid_idx.size == 0):
        return 1.0
    if (doms.ndim != 1 or doms.size != len(scopes) or len(scopes) != len(cpts) or \
            evid_idx.ndim != 1 or evid_vals.ndim != 1 or evid_idx.size != evid_vals.size or \
            (evid_idx < 0).sum() or (evid_idx >= doms.size).sum() or \
            (doms[evid_idx] - evid_vals <= 0).sum()):
        return -1.0
    # First, remove evidence from all scopes
    doms = np.array(doms)
    cpts = list(cpts)
    for i in range(len(scopes)):
        cpts[i], _ = remove_evid(scopes[i], doms[scopes[i]], cpts[i],
                                    evid_idx, evid_vals)
    doms[evid_idx] = 1
    # Next, use variable elimination in reverse-topological order to get the answer
    msg_scope = scopes[-1][-1:]
    msg_cpt, msg_doms = np.ones(doms[msg_scope].prod()), doms[msg_scope]
    for i in reversed(range(len(scopes))):
        scope1, doms1, cpt1 = scopes[i], doms[scopes[i]], cpts[i]
        marg_scope = np.setdiff1d(np.union1d(scope1, msg_scope), scope1[-1])
        msg_scope, msg_doms, msg_cpt = multiply_and_marginalize(scope1, doms1, cpt1,
                                                                     msg_scope, msg_doms, msg_cpt,
                                                                     marg_scope)
    # Finally, return the probability of evidence
    return msg_cpt.sum()

# Function to randomly assign sequence length between [2,max_seq_length] given number of samples
def gen_seqs(n, max_seq_size):
  # Check for illegal values
  if (n<1 or max_seq_size<=1):
    return np.array([], dtype=int)
  # Initialize variables
  count = 0
  seqs = []
  # Generate sequence-by-sequence
  while n-count>max_seq_size:
    seq = np.random.randint(2,max_seq_size+1)
    seqs += [ seq ]
    count += seq
  seqs += [ n-count ] if (n-count>0) else []
  # Return seqs
  return np.array(seqs, dtype=int)

# Flatten domain values to extract domain sizes
def flatten_dom_vals(dom_vals):
    doms = np.zeros(len(dom_vals), dtype=int)
    for d in range(doms.shape[0]):
        doms[d] = dom_vals[d].size
    return doms

# Learn parameters given actual domain values and scopes
def learn_model_params(dset, dom_vals, scopes, cond_scopes=[], weights=np.array([])):
    cpts = []
    # Verify if parameters are valid
    if dset.shape[1] != len(dom_vals) or (not len(cond_scopes) and len(dom_vals) != len(scopes)) or \
            weights.ndim != 1 or (len(cond_scopes) and (len(scopes) != len(cond_scopes) or \
            sum([np.setdiff1d(cond_scopes[i], scopes[i]).size for i in range(len(scopes))]) > 0)):
        return cpts
    # Initialize variables
    V = len(scopes)
    doms = flatten_dom_vals(dom_vals)
    weights = weights if (weights.size == dset.shape[0]) else np.ones(dset.shape[0])
    max_dom_size = 0
    for c in range(dset.shape[1]):
        max_dom_size = max(max_dom_size, np.unique(dset[:, c]).size)
    # smooth_factor = (dset.shape[0]*0.001) / max_dom_size
    # smooth_factor = (max(1,dset.shape[0]) * 0.001) / doms.size
    smooth_factor = Util2.SMOOTH
    # Learn CPT for each variable
    for v in range(V):
        # First, extract relevant information
        scope = scopes[v]
        cond_scope = scopes[v][-1] if not len(cond_scopes) else cond_scopes[v]
        non_cond_scope = np.setdiff1d(scope, cond_scope)
        curr_dom_vals = np.array(dom_vals)[scope]
        curr_doms = doms[scope]
        curr_assign = np.zeros(scope.shape[0], dtype=int)
        # cpt = np.zeros((curr_doms[-1],curr_doms[:-1].prod()))
        cpt = np.zeros((doms[cond_scope].prod(), doms[non_cond_scope].prod()))
        # Next, calculate the value of each cpt row
        for j in range(curr_doms[:-1].prod()):
            for i in range(curr_doms[-1]):
                matches = True
                for k in range(curr_assign.shape[0]):
                    matches *= (dset[:,scope[k]] == curr_dom_vals[k][curr_assign[k]])
                # cpt[i,j] = matches.sum() + smooth_factor
                cpt[i, j] = weights[matches].sum() + smooth_factor
                go_to_next_assign(curr_assign, curr_doms)
        # Add CPT to CPTs
        # z = cpt.sum(axis=0).reshape(1,curr_doms[:-1].prod())
        z = cpt.sum(axis=0).reshape(1, doms[non_cond_scope].prod())
        cpts += [ (cpt / z).T.flatten() ]
    # Return all CPTs
    return cpts

# Domain normalize a dataset
def dom_normalize_dset(dset, dom_vals):
  new_dset = np.array(dset)
  for c in range(dset.shape[1]):
    if (np.unique(dset[:,c]).size != dom_vals[c].size):
      return np.array([])
    dict = zip(dom_vals[c],np.arange(dom_vals[c].shape[0]))
    for k, v in dict: new_dset[dset[:,c]==k,c] = v
  return new_dset

# Remove evidence from single factor
def remove_evid(scope, doms, cpt, evid_idx, evid_vals):
    # Validation
    if(scope.ndim!=1 or cpt.ndim!=1 or doms.ndim!=1 or scope.size != doms.size or cpt.size!=doms.prod() or \
        evid_idx.ndim!=1 or evid_vals.ndim!=1 or evid_idx.size!=evid_vals.size or (evid_idx<0).sum()):
        return cpt, doms
    # Compute scope indices w.r.t. evidence
    scope_evid_idx = np.nonzero(np.in1d(scope, evid_idx))[0]
    scope_non_evid_idx = np.setdiff1d(np.arange(scope.size),scope_evid_idx)
    scope_evid_vals = evid_vals[np.hstack((np.nonzero(np.in1d(evid_idx, scope[:-1]))[0],
                                           np.nonzero(np.in1d(evid_idx, scope[-1:]))[0]))]
    # Validate if evid_vals has anything outside of the domains
    if (doms[scope_evid_idx]-scope_evid_vals<=0).sum():
        return cpt, doms
    # Initialize new_doms and new_cpt
    new_doms = np.array(doms)
    new_doms[scope_evid_idx] = 1
    new_cpt = np.zeros(new_doms.prod(), dtype=float)
    # Compute val/assignment vectors
    new_vals = np.zeros(scope_non_evid_idx.size, dtype=int)
    all_vals = np.zeros(scope.size, dtype=int)
    all_vals[scope_evid_idx] = scope_evid_vals
    # Fill in CPT
    for i in range(new_doms.prod()):
        all_vals[scope_non_evid_idx] = new_vals
        addr_all = calc_address(all_vals, doms)
        addr_new = calc_address(new_vals, new_doms[scope_non_evid_idx])
        new_cpt[addr_new] += cpt[addr_all]
        go_to_next_assign(new_vals, new_doms[scope_non_evid_idx])
    # Return new CPT and new doms
    return new_cpt, new_doms

# Multiply and marginalize two factors
def multiply_and_marginalize(scope1, doms1, factor1, scope2, doms2, factor2, marg_scope=np.array([]),
                                to_normalize=False, mpe_flag=False):
  # Validation
  if (scope1.ndim!=1 or doms1.ndim!=1 or factor1.ndim!=1 or scope1.size != doms1.size or \
      doms1.prod() != factor1.size or scope2.ndim!=1 or doms2.ndim!=1 or factor2.ndim!=1 or \
      scope2.size != doms2.size or doms2.prod() != factor2.size or marg_scope.ndim!=1 or \
      np.setdiff1d(marg_scope,np.union1d(scope1,scope2)).size>0):
      return np.array([]), np.array([]), np.array([])
  # Generate dictionaries
  scope1_dict = { key: val for key, val in zip(scope1,np.arange(len(scope1))) }
  scope2_dict = { key: val for key, val in zip(scope2,np.arange(len(scope2))) }
  # Generate joint scope
  joint_scope = np.union1d(scope1,scope2)
  joint_doms = np.array([ doms1[scope1_dict[var]] if var in scope1_dict else doms2[scope2_dict[var]] for var in joint_scope ])
  # Generate marginal scope
  marg_scope = marg_scope if marg_scope.size else joint_scope
  marg_doms = np.array([ doms1[scope1_dict[var]] if var in scope1_dict else doms2[scope2_dict[var]] for var in marg_scope ])
  marg_dict = { key: val for key, val in zip(marg_scope,np.arange(len(marg_scope))) }
  # Generate selects
  scope1_select = np.empty(scope1.shape, dtype=int)
  scope2_select = np.empty(scope2.shape, dtype=int)
  marg_select = np.empty(marg_scope.shape, dtype=int)
  for v in range(len(joint_scope)):
    var = joint_scope[v]
    if var in scope1_dict:
      scope1_select[scope1_dict[var]] = v
    if var in scope2_dict:
      scope2_select[scope2_dict[var]] = v
    if var in marg_dict:
      marg_select[marg_dict[var]] = v
  # Multiply and marginalize
  joint_assign = np.zeros(joint_scope.shape, dtype=int)
  marg_factor = np.zeros(marg_doms.prod())
  for i in range(joint_doms.prod()):
    addr1 = calc_address(joint_assign[scope1_select], doms1)
    addr2 = calc_address(joint_assign[scope2_select], doms2)
    marg_addr = calc_address(joint_assign[marg_select], marg_doms)
    if mpe_flag:
        marg_factor[marg_addr] = max(marg_factor[marg_addr], factor1[addr1] * factor2[addr2])
    else:
        marg_factor[marg_addr] += factor1[addr1] * factor2[addr2]
    go_to_next_assign(joint_assign, joint_doms)
  # Return new scope, doms and factor
  marg_factor = marg_factor / marg_factor.sum() if to_normalize and marg_factor.sum()>0.0 else marg_factor
  return marg_scope, marg_doms, marg_factor

# Marginalize a single factor over a given scope
def marginalize(scope, doms, factor, marg_scope, to_normalize=False):
  # Validation
  if (scope.ndim != 1 or doms.ndim != 1 or factor.ndim != 1 or scope.size != doms.size or \
      doms.prod() != factor.size or marg_scope.size == 0 or np.setdiff1d(marg_scope,scope).size>0):
      return scope, doms, factor
  # Create dictionary and select
  scope_dict = { key: val for key, val in zip(scope,np.arange(len(scope))) }
  marg_dict = { key: val for key, val in zip(marg_scope,np.arange(len(marg_scope))) }
  marg_doms = np.array([ doms[scope_dict[var]] for var in marg_scope ])
  marg_select = np.empty(marg_scope.shape, dtype=int)
  for v in range(len(scope)):
    var = scope[v]
    if var in marg_dict:
      marg_select[marg_dict[var]] = v
  # Marginalize
  full_assign = np.zeros(scope.shape, dtype=int)
  marg_factor = np.zeros(marg_doms.prod())
  for i in range(doms.prod()):
    addr = calc_address(full_assign, doms)
    marg_addr = calc_address(full_assign[marg_select], marg_doms)
    marg_factor[marg_addr] += factor[addr]
    go_to_next_assign(full_assign, doms)
  # Return new scope, doms and factor
  marg_factor = marg_factor / marg_factor.sum() if to_normalize and marg_factor.sum()>0.0 else marg_factor
  return marg_scope, marg_doms, marg_factor

# Returns the address of the entry of a given factor table that has the highest value
def factor_argmax(factor, doms):
  if doms.ndim != 1 or factor.ndim != 1 or factor.size != doms.prod():
    return -1
  addr = np.argmax(factor)
  return calc_vals_from_address(addr, doms)

# Generate all possible edges of a complete graph with V vertices
def gen_edges_ij(V=1):
    if V <= 0:
        return np.array([]), np.array([])
    v_idx = np.arange(V)
    all_edges = np.array(list(combinations(v_idx, 2)))
    return all_edges[:, 0], all_edges[:, 1]

# Generate partitions from MI matrix by removing weak edges based on a
# dynamic adaptive threshold (1/n)^b
def gen_adaptive_partitions(mi_matrix, n=1, b=0.5):
  # Validation
  if mi_matrix.ndim !=2 or mi_matrix.shape[0] != mi_matrix.shape[1] or \
    n<1 or b<=0.0 or b>=1.0:
    return []
  # Calculate threshold and remove weak edges
  threshold = (1.0/n) ** b
  copy_matrix = mi_matrix.copy()
  copy_matrix[mi_matrix<=threshold] = 0.0
  # Generate minimum spanning tree
  X = csr_matrix(-copy_matrix)
  T = minimum_spanning_tree(X)
  topo_order, _ = depth_first_order(T,0,directed=False)
  partitions = [ np.sort(topo_order) ]
  # Add forest components to partitions
  V = np.arange(copy_matrix.shape[0])
  while topo_order.size < V.size:
    v = np.setdiff1d(V, topo_order)[0]
    v_topo_order, _ = depth_first_order(T, v, directed=False)
    topo_order = np.append(topo_order, v_topo_order)
    partitions += [ np.sort(v_topo_order) ]
  # Return partitions
  return partitions

# Generate partitions from MI matrix by using clustering
def gen_clustered_partitions(mi_matrix, C=0, L=1):
  # Validation
  if mi_matrix.ndim !=2 or mi_matrix.shape[0] != mi_matrix.shape[1]:
    return []
  if C<=0 or L==1:
    return [ np.array([v]) for v in range(mi_matrix.shape[0]) ]
  # Compute cluster centers
  V = mi_matrix.shape[0]
  centers = np.array([ 0 ])
  while len(centers)<C:
      other_vars = np.setdiff1d(np.arange(V),centers)
      all_scores = mi_matrix[centers][:,other_vars]
      center_scores = all_scores.mean(axis=0) - all_scores.std(axis=0)
      new_center = other_vars[np.argmin(center_scores)]
      centers = np.union1d(centers, new_center)
  # Sort into clusters
  other_vars = np.setdiff1d(np.arange(V),centers)
  clusters = [ np.array([c]) for c in centers ]
  center_to_cluster_map = dict(zip(centers,np.arange(centers.size)))
  for var in other_vars:
      center_index = np.argmax(mi_matrix[var][centers])
      center = centers[center_index]
      c = center_to_cluster_map[center]
      clusters[c] = np.union1d(clusters[c], var)
      if clusters[c].size >= L:
          centers = np.delete(centers, center_index)
  # Return partitions
  return clusters

# Generate partitions from MI Matrix by recursively applying the CLT algorithm
def gen_clt_partitions(mi_matrix, L, idx_replace):
  # Check if L condition is satisfied
  if mi_matrix.shape[0] <= L:
    return [ idx_replace[np.arange(mi_matrix.shape[0])] ]
  # Learn maximum spanning tree
  X = csr_matrix(-mi_matrix)
  T = minimum_spanning_tree(X)
  topo_order, parents = depth_first_order(T,0,directed=False)
  edges_i, edges_j = parents[topo_order[1:]], topo_order[1:]
  # Remove weakest edge
  weakest_idx = np.argmin(mi_matrix[edges_i,edges_j])
  T[edges_i[weakest_idx],edges_j[weakest_idx]] = 0.0
  T[edges_j[weakest_idx],edges_i[weakest_idx]] = 0.0
  T.eliminate_zeros()
  # Split into two sub-trees
  topo_order_i = depth_first_order(T,edges_i[weakest_idx],directed=False)[0]
  topo_order_j = depth_first_order(T,edges_j[weakest_idx],directed=False)[0]
  topo_order_i.sort()
  topo_order_j.sort()
  idx_replace_i = idx_replace[topo_order_i]
  idx_replace_j = idx_replace[topo_order_j]
  # Recursively partition clusters
  clusters = gen_clt_partitions(mi_matrix[topo_order_i][:,topo_order_i],L,idx_replace_i) + \
              gen_clt_partitions(mi_matrix[topo_order_j][:,topo_order_j],L,idx_replace_j)
  # Return clusters
  return clusters
