import numpy as np
from ultility import *

variable_value_list = {}

class Node(object):
    """Node in a computation graph."""

    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = add_op(self, constant(other))
        return new_node

    def __sub__(self, other):
        """Subtracting two nodes return a new node."""
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = sub_op(self, constant(other))
        return new_node

    def __rsub__(self, other):
        """Subtracting two nodes return a new node."""
        if isinstance(other, Node):
            new_node = sub_op(other, self)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = sub_op(constant(other), self)
        return new_node

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_op(self, constant(other))
        return new_node

    def __div__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = div_op(self, constant(other))
        return new_node

    def __rdiv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(other, self)
        else:
            new_node = div_op(constant(other), self)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    # I don't know what they are
    __floordiv__ = __div__
    __rfloordiv__ = __rdiv__

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __neg__(self):
        """-node_self, return a new node."""
        new_node = constant(0) - self
        return new_node

    def __str__(self):
        """Allow print to display node name."""
        return self.name

    __repr__ = __str__


class Op(object):
    """Op represents operations performed on nodes."""

    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [adapt(output_grad, node.inputs[0]), adapt(output_grad, node.inputs[1])]


class SubOp(Op):
    """Op to element-wise sub two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [adapt(output_grad, node.inputs[0]), adapt(output_grad, node.inputs[1])]

class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        return [adapt(output_grad * node.inputs[1], node.inputs[0]),
                adapt(output_grad * node.inputs[0], node.inputs[1])]


class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise division."""
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of divide node, return gradient contributions to each input."""
        grad_A = output_grad / node.inputs[1]
        grad_B = -1 * output_grad * node.inputs[0] / node.inputs[1] / node.inputs[1]
        return [adapt(grad_A, node.inputs[0]), adapt(grad_B, node.inputs[1])]


class MatMulOp(Op):
    """Op to matrix multiply two nodes."""

    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 2
        if node.matmul_attr_trans_A:
            A = np.transpose(input_vals[0])
        else:
            A = input_vals[0]
        if node.matmul_attr_trans_B:
            B = np.transpose(input_vals[1])
        else:
            B = input_vals[1]
        return np.matmul(A, B)

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.

        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        """TODO: Your code here"""
        boolA = node.matmul_attr_trans_A
        boolB = node.matmul_attr_trans_B
        gradA = matmul(output_grad, node.inputs[1], boolA, not boolB)
        gradB = matmul(node.inputs[0], output_grad, not boolA, boolB)
        return [gradA, gradB]


class LogOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "log(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad / node.inputs[0]]


class ExpOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "exp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * exp(node.inputs[0])]

class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""

    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert (isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""

    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        # assert (isinstance(input_vals[0], np.ndarray))
        assert len(input_vals) == 1
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class ReduceSumOp(Op):
    """Op to compute reduce sum"""
    def __call__(self, node_A, axis=None, reduction_indices=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if reduction_indices is not None:
            assert axis is None
            axis = tuple(reduction_indices)
        new_node.const_attr = axis
        new_node.name = "Reducesum(%s, axis=%s)" % (node_A.name, str(axis))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sum(input_vals[0], axis=node.const_attr)

    def gradient(self, node, output_grad):
        return [broadcast_to(output_grad, node.inputs[0])]


class ReduceMeanOp(Op):
    """Op to compute reduce mean"""
    def __call__(self, node_A, axis=None, reduction_indices=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if reduction_indices is not None:
            assert axis is None
            axis = tuple(reduction_indices)
        new_node.const_attr = axis
        new_node.name = "Reducemean(%s, axis=%s)" % (node_A.name, str(axis))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.mean(input_vals[0], axis=node.const_attr)

    def gradient(self, node, output_grad):
        return [broadcast_to(output_grad, node.inputs[0]) / reduce_sum(node.inputs[0])]


class ReluOp(Op):
    """Op to compute reduce mean"""
    def __call__(self, node_A, name=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if name is None: new_node.name = "relu(%s)" % node_A.name
        else: new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.maximum(input_vals[0], 0)

    def gradient(self, node, output_grad):
        assert False


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""

    def __call__(self, dtype, shape=None, name="placeholder"):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        new_node.const_attr = (shape, dtype)
        new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None


class VariableOp(Op):
    """Op that represents variables"""
    def __call__(self, init, dtype=None, shape=None, name="Variables"):
        new_node = Op.__call__(self)
        new_node.name = name
        if dtype is None:
            variable_value_list[new_node] = init
        else:
            '''
            if isinstance(init, np.ndarray):
                variable_value_list[new_node] = init.astype(dtype)
            else:
            '''
            variable_value_list[new_node] = np.array(init).astype(dtype)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 0
        assert not(node.const_attr is None)
        return node.const_attr

    def gradient(self, node, output_grads):
        return None


class ConstantOp(Op):
    """ Op that represents a constant. """
    def __call__(self, val, name="Const"):
        new_node = Op.__call__(self)
        new_node.const_attr = np.array(val)
        new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        return node.const_attr

    def gradient(self, node, output_grad):
        return None


class AssignOp(Op):
    """ Op that assigns value to a node. """
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        if not isinstance(node_B, Node):
            node_B = constant(node_B)
        new_node.const_attr = node_A
        new_node.inputs = [node_B]
        new_node.name = "Assign(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert isinstance(node.const_attr.op, VariableOp)
        node.const_attr.const_attr = input_vals[0]
        return input_vals[0]


class BroadcastToOp(Op):
    """Op that broadcasts value of node_A to the shape of node_B."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastTo(%s, %s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        # return np.broadcast_to(input_vals[0], input_vals[1].shape)
        output_val = input_vals[0]
        # not complete yet
        if len(output_val.shape) < len(input_vals[1].shape):
            front_align = True
            for dim, in_size in enumerate(output_val.shape):
                if input_vals[1].shape[dim] != in_size:
                    front_align = False
                    break
            new_shape = output_val.shape
            if front_align:
                while len(new_shape) < len(input_vals[1].shape):
                    new_shape = new_shape + (1,)
            output_val.resize(new_shape)
        output_val = np.broadcast_to(output_val, input_vals[1].shape)
        return output_val

    def gradient(self, node, output_grad):
        return [adapt(output_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]


class EqualOp(Op):
    """ Op that computes whether two nodes are equal."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s == %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return np.equal(input_vals[0], input_vals[1])


class ArgmaxOp(Op):
    def __call__(self, node_A, axis=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = axis
        new_node.name = "Argmax(%s, axis=%s)" % (node_A.name, str(axis))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.argmax(input_vals[0], axis=node.const_attr)


class ShapeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Shape(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.shape(input_vals[0])


class AdaptOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "Adapt(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        '''
        assert len(input_vals) == 2
        flag = find(input_vals[0].shape, input_vals[1].shape)
        assert flag != 0
        if flag == -1:
            cur = input_vals[0]
            while len(cur.shape) > len(input_vals[1].shape):
                cur = cur[0]
            return cur
        else:
            cur = input_vals[0]
            while len(cur.shape) > len(input_vals[1].shape):
                cur = np.mean(cur, axis=-1)
            return cur
        '''
        assert len(input_vals) == 2

        output_val = input_vals[0]
        while len(output_val.shape) > len(input_vals[1].shape):
            output_val = np.sum(output_val, axis=0)
        for dim in range(len(output_val.shape)):
            if output_val.shape[dim] > input_vals[1].shape[dim]:
                assert input_vals[1].shape[dim] == 1
                output_val = np.sum(output_val, axis=dim, keepdims=True)

        return output_val

    def gradient(self, node, output_grad):
        return [broadcast_to(output_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]


class CastOp(Op):
    def __call__(self, node_A, dtype, name=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = dtype
        if name is None: new_node.name = "Cast(%s, dtype=%s)" % (node_A.name, dtype)
        else: new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0].astype(node.const_attr)


class WrapOp(Op):
    def __call__(self, node_list):
        new_node = Op.__call__(self)
        new_node.inputs = node_list
        new_node.name = "Wrap(%s)" % (str(node_list))
        return new_node

    def compute(self, node, input_vals):
        return None


class VariablesInitOp(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        new_node.inputs = []
        new_node.name = "Global_Variables_Initializer"
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 0
        for key, value in variable_value_list.items():
            if isinstance(value, Node):
                key.const_attr = value.const_attr
            else:
                key.const_attr = value


# Create global singletons of operators.
add_op = AddOp()
sub_op = SubOp()
mul_op = MulOp()
div_op = DivOp()
matmul = MatMulOp()
log = LogOp()
exp = ExpOp()
zeroslike_op = ZerosLikeOp()
oneslike_op = OnesLikeOp()
reduce_sum = ReduceSumOp()
reduce_mean = ReduceMeanOp()
relu_op = ReluOp()
placeholder = PlaceholderOp()
Variable = VariableOp()
constant = ConstantOp()
assign = AssignOp()
broadcast_to = BroadcastToOp()
equal = EqualOp()
argmax = ArgmaxOp()
shape = ShapeOp()
adapt = AdaptOp()
cast = CastOp()
wrap = WrapOp()
global_variables_initializer = VariablesInitOp()