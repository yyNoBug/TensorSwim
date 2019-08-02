import ultility as ut
from neuralnetwork import *

class Session(object):
    def __call__(self, name="Session"):
        newSession = Session()
        newSession.name = name
        newSession.ex = None   # I don't know what it means
        return newSession

    def run(self, eval_node_list, feed_dict={}):
        if isinstance(eval_node_list, list):
            executor = Executor(eval_node_list)
        else:
            executor = Executor([eval_node_list])
        return executor.run(feed_dict=feed_dict)[0]

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        return

class Executor:
    """Executor computes values for a given subset of nodes in a computation graph."""

    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list.
        """
        node_to_val_map = dict(feed_dict)
        for key, value in node_to_val_map.items():
            node_to_val_map[key] = np.array(value)
        # Traverse graph in topological sort order and compute values for all nodes.
        topo_order = ut.find_topo_sort(self.eval_node_list)
        """TODO: Your code here"""
        for n in topo_order:
            if n in node_to_val_map: continue
            feed = ut.search(n.inputs, node_to_val_map)
            node_to_val_map[n] = n.op.compute(n, feed)

        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(ut.find_topo_sort([output_node]))

    """TODO: Your code here"""
    for n in reverse_topo_order:

        grad = ut.sum_node_list(node_to_output_grads_list[n])
        node_to_output_grad[n] = grad

        input_grads = n.op.gradient(n, grad)
        for i in range(len(n.inputs)):
            if n.inputs[i] in node_to_output_grads_list:
                node_to_output_grads_list[n.inputs[i]].append(input_grads[i])
            else:
                node_to_output_grads_list[n.inputs[i]] = [input_grads[i]]

    # node_to_output_grad 只储存节点
    # node_to_output_grads_list 节点从各个output下传的导数节点

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list