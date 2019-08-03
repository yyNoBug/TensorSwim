import ultility as ut
from functional import *



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
        count = 0
        for n in topo_order:
            if n in node_to_val_map: continue
            print(count)
            count += 1
            feed = ut.search(n.inputs, node_to_val_map)
            node_to_val_map[n] = n.op.compute(n, feed)

        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results


