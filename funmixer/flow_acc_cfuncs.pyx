"""
This module contains functions for building a topological stack of nodes in a flow direction array.
For speed and memory efficiency, the functions are written in Cython. Sample graph construction algorithm
initially developed by Rich Barnes in C++ and translated to Cython by Alex Lipp.
"""
# distutils: language = c++
from libcpp.stack cimport stack
from libcpp.queue cimport queue
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp
cimport cython

ROOT_NODE_NAME = "##ROOT##"
UNSET_NODE_NAME = "##UNSET##"
NO_DOWNSTREAM_NEIGHBOUR = 2**32 - 1  # Maximum value for a 64-bit integer, used to indicate no downstream neighbour

cdef class SampleData:
    """
    A class to hold sample site data (name and coordinates) for a node in the sample graph.
    """
    cdef public str name
    cdef public cnp.float64_t x
    cdef public cnp.float64_t y

    def __cinit__(self):
        self.name = UNSET_NODE_NAME
        self.x = -1.
        self.y = -1.

    # Define a factory method to create a SampleData instance from a dictionary
    @staticmethod
    def from_dict(data):
        cdef SampleData sample = SampleData()
        sample.name = data.get("name", UNSET_NODE_NAME)
        sample.x = data.get("x", -1.)
        sample.y = data.get("y", -1.)
        return sample

cdef class NativeSampleNode:
    """
    A class to hold a node in the sample graph. This will be converted into a Python object later.
    """
    cdef public SampleData data
    cdef public cnp.int64_t downstream_node
    cdef public list upstream_nodes
    cdef public cnp.int64_t area
    cdef public cnp.int64_t total_upstream_area
    cdef public cnp.int64_t label

    def __cinit__(self):
        self.data = SampleData()
        self.downstream_node = NO_DOWNSTREAM_NEIGHBOUR
        self.upstream_nodes = []
        self.area = 0
        self.total_upstream_area = 0
        self.label = -1  # Default label, will be set later

    @staticmethod
    def make_root_node():
        """
        Factory method to create a root node with a default label and name.
        """
        cdef NativeSampleNode temp
        temp = NativeSampleNode()
        temp.label = 0  # Root node label
        temp.data.name = ROOT_NODE_NAME
        return temp

    @staticmethod
    def make_w_downstream_and_sample(cnp.int64_t downstream_node, SampleData sample_data):
        """
        Factory method to create a node with a downstream neighbour and sample data.
        """
        cdef NativeSampleNode temp = NativeSampleNode()
        temp.downstream_node = downstream_node
        temp.data = sample_data
        return temp

### The following functions are related to the D8 flow direction array and its processing ###

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

### The following algorithms are related to the sample graph and its upstream areas ###

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calculate_total_upstream_areas(list sample_graph):
    """Calculates the total upstream areas for each sample node in the sample graph.
    Args:
        sample_graph: A list of NativeSampleNode objects representing the sample graph.
        (Note: The list is expected to contain NativeSampleNode objects.)
    """
    cdef int n = len(sample_graph)
    # Count how many upstream neighbours we're waiting on
    cdef int[:] deps = np.zeros(n, dtype=np.int32)
    cdef int i, c
    cdef NativeSampleNode self

    # Initialize the dependencies array
    for i in range(n):
        deps[i] = len(sample_graph[i].upstream_nodes)

    # Find cells with no upstream neighbours
    cdef queue[int] q
    for i in range(n):
        if deps[i] == 0:
            q.push(i)

    while not q.empty():
        c = q.front()  # Get the front of the queue
        q.pop()  # Remove the front element from the queue
        self = sample_graph[c]  # Get the current sample node
        # Add my own area
        self.total_upstream_area += self.area

        if self.downstream_node == NO_DOWNSTREAM_NEIGHBOUR:
            continue  # If no downstream neighbour, skip to the next iteration

        # Add my area to downstream neighbour
        sample_graph[self.downstream_node].total_upstream_area += self.total_upstream_area

        # My downstream node no longer depends on me
        deps[self.downstream_node] -= 1

        if deps[self.downstream_node] == 0:
            q.push(self.downstream_node)  # If no more dependencies, add to the queue


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def build_samplesite_graph(
    cnp.int64_t[:] receivers, 
    cnp.ndarray[cnp.int64_t, ndim=1] baselevel_nodes, 
    dict[int, dict[str,object]] sample_dict,
    ):
    """
    Creates a map of subbasins, where each subbasin is assigned a unique ID from 0 (baselevel) to n (number of subbasins). 
    Each subbasin has at its mouth a sample site node. 

    Args: 
        receivers: The receiver array (i.e., receiver[i] is the ID
        of the node that receives the flow from the i'th node).
        baselevel_nodes: The baselevel nodes to start from (i.e., sink-nodes).
        sample_dict: A dictionary mapping sample site nodes to their metadata,
        where keys are node indices and metadata is a dictionary with keys "name", "x", and "y".
    """
    # Initialize variables
    cdef int nr = len(receivers)
    cdef int[:] n_donors = count_donors(receivers)
    cdef int[:] delta = ndonors_to_delta(n_donors)
    cdef int[:] donors = make_donor_array(receivers, delta)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] labels = np.zeros(nr, dtype=np.int64)
    cdef int node, m
    cdef cnp.int64_t my_new_label = 0
    cdef cnp.int64_t my_current_label
    cdef NativeSampleNode parent
    cdef SampleData data
    # Initialize the sample parent graph as a list of NativeSampleNode
    cdef list sample_parent_graph = []
    # Initialize a queue for breadth-first search
    cdef queue[int] q
    # Push a root node into the list
    sample_parent_graph.append(NativeSampleNode.make_root_node())

    # Add baselevel nodes to the queue
    for b in baselevel_nodes:
        q.push(b)

    # Perform a breadth-first search to label subbasins
    while not q.empty():
        # Get the node from the front of the queue
        node = q.front()
        q.pop()

        # Check if the node is a sample site
        if node in sample_dict:
            # Extract sample data from the sample_dict
            data = SampleData.from_dict(sample_dict[node])
            # Create a new sample node with the sample data
            my_new_label = len(sample_parent_graph)
            my_current_label = labels[node]
            parent = sample_parent_graph[my_current_label]
            parent.upstream_nodes.append(data.name)  # Add the sample site name to the parent node's upstream nodes
            sample_parent_graph.append(
                NativeSampleNode.make_w_downstream_and_sample(my_current_label, data)
            )  # Create a new sample node with the sample data
            # Update the label for the sample site node
            labels[node] = my_new_label  # Assign the new label to the sample site node

        # Get the label of the current node
        my_label = labels[node]
        sample_parent_graph[my_label].area += 1  # Increment the area of the sample node

        # Loop through the donors of the node and add them to the queue
        for n in range(delta[node], delta[node + 1]):
            m = donors[n]
            if m != node:
                labels[m] = my_label  # Assign the label of the node to the donor
                q.push(m)  # Add the donor to the queue

    # Calculate the total upstream areas for each sample node having built the sample parent graph
    calculate_total_upstream_areas(sample_parent_graph)
    # Check that the total upstream area of the root node is equal to the number of pixels
    if sample_parent_graph[0].total_upstream_area != nr:
        raise ValueError("Total upstream area of root node does not match the number of pixels in the flow direction array!")

    # Loop through the sample parent graph to assign labels
    for i in range(len(sample_parent_graph)):
        # Assign the label to the sample parent node
        sample_parent_graph[i].label = i  # Assign the label to the sample parent node

    return labels, sample_parent_graph