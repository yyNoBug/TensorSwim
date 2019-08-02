from operation import *


class nn(object):
    class SoftmaxOp(Op):
        def __call__(self, node_A):
            tmp = exp(node_A)
            new_node = tmp / reduce_sum(tmp, axis=1)
            new_node.name = "Softmax(%s)" % node_A.name
            return new_node

    softmax = SoftmaxOp()