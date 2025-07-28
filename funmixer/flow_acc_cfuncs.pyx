"""
This module contains functions for building a topological stack of nodes in a flow direction array.
For speed and memory efficiency, the functions are written in Cython.
"""
# distutils: language = c++
from libcpp.stack cimport stack
from libcpp.queue cimport queue
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
def d8_to_receivers(cnp.ndarray[cnp.int64_t, ndim=2] arr) -> cnp.int64_t[:]:
    """
    Converts a D8 flow direction array to a receiver array.

    Args:
        arr: A D8 flow direction array.

    Returns:
        A receiver array.
    """
    cdef Py_ssize_t nrows = arr.shape[0]
    cdef Py_ssize_t ncols = arr.shape[1]
    cdef cnp.int64_t[:] receivers = np.empty(nrows * ncols, dtype=np.int64)
    cdef Py_ssize_t i, j
    cdef Py_ssize_t cell

    for i in range(nrows):
        for j in range(ncols):
            cell = i * ncols + j
            # Check if boundary cell
            if i == 0 or j == 0 or i == nrows - 1 or j == ncols - 1 or arr[i, j] == 0:
                receivers[cell] = cell
            elif arr[i, j] == 1:  # Right
                receivers[cell] = i * ncols + j + 1
            elif arr[i, j] == 2:  # Lower right
                receivers[cell] = (i + 1) * ncols + j + 1
            elif arr[i, j] == 4:  # Bottom
                receivers[cell] = (i + 1) * ncols + j
            elif arr[i, j] == 8:  # Lower left
                receivers[cell] = (i + 1) * ncols + j - 1
            elif arr[i, j] == 16:  # Left
                receivers[cell] = i * ncols + j - 1
            elif arr[i, j] == 32:  # Upper left
                receivers[cell] = (i - 1) * ncols + j - 1
            elif arr[i, j] == 64:  # Top
                receivers[cell] = (i - 1) * ncols + j
            elif arr[i, j] == 128:  # Upper right
                receivers[cell] = (i - 1) * ncols + j + 1
            else:
                raise ValueError(f"Invalid flow direction value: {arr[i, j]}")
    return receivers


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def count_donors(cnp.int64_t[:] r) -> int[:] :
    """
    Counts the number of donors that each cell has.

    Args:
        r: The receiver indices.

    Returns:
        An array of donor counts.
    """
    cdef int n = len(r)  # np = number of pixels
    cdef int[:] d = np.zeros(n, dtype=np.int32)
    cdef int j
    for j in range(n):
        d[r[j]] += 1
    return d    

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ndonors_to_delta(int[:] nd) -> int[:] :
    """
    Converts a number of donors array to an index array that contains the location of where the list of
    donors to node i is stored.

    Args:
        nd: The donor array.

    Returns:
        An array of donor counts.
    """
    cdef int n = len(nd)
    # Initialize the index array to the number of pixels
    cdef int[:] delta = np.zeros(n + 1, dtype=np.int32)
    delta[n] = n
    cdef int i
    for i in range(n, -1, -1):
        if i == n:
            continue
        delta[i] = delta[i + 1] - nd[i]

    return delta    

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def make_donor_array(cnp.int64_t[:] r, int[:] delta) -> int[:] :
    """
    Makes the array of donors. This is indexed according to the delta
    array. i.e., the donors to node i are stored in the range delta[i] to delta[i+1].
    So, to extract the donors to node i, you would do:
    donors[delta[i]:delta[i+1]]

    Args:
        r: The receiver indices.
        delta: The delta index array.

    Returns:
        The donor array.
    """
    cdef int n = len(r)  # np = number of pixels
    # Define an integer working array w intialised to 0.
    cdef int[:] w = np.zeros(n, dtype=np.int32)
    # Donor array D
    cdef int[:] D = np.zeros(n, dtype=np.int32)
    cdef int i
    for i in range(n):
        D[delta[r[i]] + w[r[i]]] = i
        w[r[i]] += 1

    return D    

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def build_ordered_list_iterative(cnp.int64_t[:] receivers, cnp.ndarray[cnp.int64_t, ndim=1] baselevel_nodes) -> int[:] :
    """
    Builds the ordered list of nodes in topological order, given the receiver array.
    Starts at the baselevel nodes and works upstream in a wave building a 
    breadth-first search order of the nodes using a queue. This is much faster
    than the recursive version. 

    Args:
        receivers: The receiver array (i.e., receiver[i] is the ID
        of the node that receives the flow from the i'th node).
        baselevel_nodes: The baselevel nodes to start from.

    Returns:
        The nodes in topological order (using a BFS).
    """
    cdef int n = len(receivers)
    cdef int[:] n_donors = count_donors(receivers)
    cdef int[:] delta = ndonors_to_delta(n_donors)
    cdef int[:] donors = make_donor_array(receivers, delta)
    cdef int[:] ordered_list = np.zeros(n, dtype=np.int32) - 1
    cdef int j = 0 # The index in the stack (i.e., topological order)
    cdef int b, node, m
    # Queue for breadth-first search
    cdef queue[int] q

    # Add baselevel nodes to the stack
    for b in baselevel_nodes:
        q.push(b)  # Add the baselevel node to the queue

    while not q.empty():
        node = q.front()  # Get the node from the front of the queue
        q.pop()  # Remove the node from the queue
        ordered_list[j] = node # Add the node to the stack
        j += 1 # Increment the stack index.
        # Loop through the donors of the node
        for n in range(delta[node], delta[node+1]):
            m = donors[n]
            if m != node:
                q.push(m)  # Add the donor to the queue
    return ordered_list

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def build_samplesite_graph(
    cnp.int64_t[:] receivers, 
    cnp.ndarray[cnp.int64_t, ndim=1] baselevel_nodes, 
    cnp.ndarray[cnp.int64_t, ndim=1] samplesite_nodes
    ):
    """
    Creates a map of subbasins, where each subbasin is assigned a unique ID from 0 (baselevel) to n (number of subbasins). 
    Each subbasin has at its mouth a sample site node. 

    Args: 
        receivers: The receiver array (i.e., receiver[i] is the ID
        of the node that receives the flow from the i'th node).
        baselevel_nodes: The baselevel nodes to start from (i.e., sink-nodes).
        samplesite_nodes: The sample site nodes to assign to each subbasin.

    Returns:
        A map of subbasins, where each subbasin is assigned a unique ID from 0 (baselevel) to n (number of subbasins).
    """
    # Initialize variables
    cdef int n = len(receivers)
    cdef int[:] n_donors = count_donors(receivers)
    cdef int[:] delta = ndonors_to_delta(n_donors)
    cdef int[:] donors = make_donor_array(receivers, delta)
    cdef cnp.ndarray[double, ndim=1] labels = np.zeros(n, dtype=np.float64)  # Initialize labels to 0
    cdef int label_value = 0  # Start labeling from 0
    cdef int node, m

    # Create unordered_set for samplesite_nodes to check membership efficiently
    cdef unordered_set[cnp.int64_t] samplesite_set
    for i in range(samplesite_nodes.shape[0]):
        samplesite_set.insert(samplesite_nodes[i])

    # Initialize queue for breadth-first search
    cdef queue[int] q

    # Add baselevel nodes to the queue
    for b in baselevel_nodes:
        q.push(b)

    # Perform a breadth-first search to label subbasins
    while not q.empty():
        # Get the node from the front of the queue
        node = q.front()
        q.pop()

        # Check if the node is a sample site
        if samplesite_set.count(node) != 0:
            label_value += 1  # Increment the label value for the next sample site
            labels[node] = label_value  # Assign the new label to the sample site node
        else:
            # If the node is not a sample site, assign it the label of its receiver
            labels[node] = labels[receivers[node]] if receivers[node] != node else 0

        # Get the label of the current node
        my_label = labels[node]

        # Loop through the donors of the node and add them to the queue
        for n in range(delta[node], delta[node + 1]):
            m = donors[n]
            if m != node:
                labels[m] = my_label  # Assign the label of the node to the donor
                q.push(m)  # Add the donor to the queue

    return labels

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def accumulate_flow(
    cnp.int64_t[:] receivers,
    int[:] ordered, 
    cnp.ndarray[double, ndim=1] weights
):
    """
    Accumulates flow along the stack of nodes in topological order, given the receiver array,
    the ordered list of nodes, and a weights array which contains the contribution from each node.

    Args:
        receivers: The receiver array (i.e., receiver[i] is the ID
        of the node that receives the flow from the i'th node).
        ordered: The ordered list of nodes.
        weights: The weights array (i.e., the contribution from each node).
    """
    cdef int n = receivers.shape[0]
    cdef cnp.ndarray[double, ndim=1] accum = weights.copy()
    cdef int i
    cdef cnp.int64_t donor, recvr

    # Accumulate flow along the stack from upstream to downstream
    for i in range(n - 1, -1, -1):
        donor = ordered[i]
        recvr = receivers[donor]
        if donor != recvr:
            accum[recvr] += accum[donor]

    return accum