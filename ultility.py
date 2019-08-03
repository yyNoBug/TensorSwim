def search(list, dic):
    """Search for listed items in dic and return them."""
    ans = []
    for i in list:
        if i in dic: ans.append(dic[i])
    return ans


def search_ind(list, dic):
    """Search for listed items in dic and return them."""
    ans = []
    for i in list:
        if i in dic: ans.append(i)
    return ans


def find(a, b):
    """Find whether tuple a is in tuple b.
        Return 1 when a is the head of b
        Return -1 when a is the rear of b
        Return 0 for any other cases.
    """
    flag = True
    for i in range(len(a)):
        if a[i] != b[i]:
            flag = False
            break
    if flag: return 1
    for i in range(len(a)):
        if a[i] != b[-1-i]: return 0
    return -1


def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)