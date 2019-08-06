from operation import *
from ultility import *


class nn(object):
    class SoftmaxOp(Op):
        def __call__(self, node_A):
            tmp = exp(node_A)
            new_node = tmp / broadcast_to(reduce_sum(tmp, axis=-1), tmp)
            new_node.name = "Softmax(%s)" % node_A.name
            return new_node

    class SoftmaxCrossEntropyWithLogitsOp(Op):
        def __call__(self, logits, labels):
            y_pred = nn.softmax(logits)
            new_node = -reduce_sum(labels * log(y_pred), -1)
            new_node.name = "SCEWL(%s)" %logits.name
            return new_node

    class ReluOp(Op):
        def __call__(self, features, name=None):
            new_node = relu_op(features, name=None)
            return new_node

    class Conv2dOp(Op):
        def __call__(self, input, filter, strides, padding):
            new_node = conv2d_op(input, filter, strides, padding)
            return new_node

    class MaxPoolOp(Op):
        def __call__(self, value, ksize, strides, padding):
            new_node = max_pool_op(value, ksize, strides, padding)
            return new_node

    softmax = SoftmaxOp()
    softmax_cross_entropy_with_logits = SoftmaxCrossEntropyWithLogitsOp()
    relu = ReluOp()
    conv2d = Conv2dOp()
    max_pool = MaxPoolOp()


class train(object):
    class GradientDescentOptimizer():
        def __init__(self, learning_rate=0.01, name="GradientDescent"):
            self.learning_rate = learning_rate
            self.name = name

        def minimize(self, loss):
            valid_nodes = find_topo_sort([loss])
            variables_optimize = search_ind(valid_nodes, variable_value_list)
            variables_gradients = gradients(loss, variables_optimize)
            new_node_list = []
            for i, item in enumerate(variables_optimize):
                new_node_list.append(assign(item, item - variables_gradients[i] * self.learning_rate))
            return wrap(new_node_list)

    class AdamOptimizer():
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam'):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.name = name
            self.t = 0
            self.m = []
            self.v = []

        def minimize(self, loss):
            valid_nodes = find_topo_sort([loss])
            variables_optimize = search_ind(valid_nodes, variable_value_list)
            variables_gradients = gradients(loss, variables_optimize)
            new_node_list = []

            for _ in variables_optimize:
                self.m.append(0)
                self.v.append(0)

            self.t = self.t + 1
            lr_t = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t)

            for ind in range(len(variables_gradients)):
                m_t = self.beta1 * self.m[ind] + (1 - self.beta1) * variables_gradients[ind]
                v_t = self.beta2 * self.v[ind] + (1 - self.beta2) * variables_gradients[ind] * variables_gradients[ind]
                variable = variables_optimize[ind] - lr_t * m_t / (sqrt(v_t) + self.epsilon)
                new_node_list.append(assign(variables_optimize[ind], variable))

            return wrap(new_node_list)


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
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    """TODO: Your code here"""
    for n in reverse_topo_order:

        grad = sum_node_list(node_to_output_grads_list[n])
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