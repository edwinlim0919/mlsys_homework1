from typing import Any, Dict, List

import numpy as np


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[np.ndarray]
            The input values of the given node.

        Returns
        -------
        output: np.ndarray
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]
        #curr_sum = input_values[0] + input_values[1]
        #if len(input_values) > 2:
        #    for inp in input_values[2:]:
        #        curr_sum = curr_sum + inp

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of input values."""
        """TODO: Your code here"""
        assert len(input_values) == 2
        if not input_values:
            raise ValueError('None')

        result = np.prod(input_values, axis=0)
        return result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        """TODO: Your code here"""
        return [output_grad*node.inputs[1], output_grad*node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of the input value and the constant."""
        """TODO: Your code here"""
        if not input_values or not node:
            raise ValueError('None')

        const = node.__getattr__('constant')
        result = np.squeeze([array * const for array in input_values])
        return result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        """TODO: Your code here"""
        const = node.__getattr__('constant')
        return [output_grad * const]



class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of input values."""
        """TODO: Your code here"""
        assert len(input_values) == 2
        if not input_values:
            raise ValueError('None')

        result = input_values[0].astype(float)
        for arr in input_values[1:]:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(result, arr, where=arr!=0)

        return result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        """TODO: Your code here"""
        return [output_grad/node.inputs[1], (-1 * node.inputs[0] / (node.inputs[1] * node.inputs[1])) * output_grad]



class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of the input value and the constant."""
        """TODO: Your code here"""
        if not input_values or not node:
            raise ValueError('None')

        const = float(node.__getattr__('constant'))
        stacked_inputs = np.stack(input_values, axis=0)
        if const != 0.0:
            result = np.squeeze(stacked_inputs / const)
        else:
            return np.array([])

        return result


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        """TODO: Your code here"""
        const = float(node.__getattr__('constant'))
        return [output_grad / const]


class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node, trans_A: bool = False, trans_B: bool = False
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix
        trans_A: bool
            A boolean flag denoting whether to transpose A before multiplication.
        trans_B: bool
            A boolean flag denoting whether to transpose B before multiplication.

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"trans_A": trans_A, "trans_B": trans_B},
            name=f"({node_A.name + ('.T' if trans_A else '')}@{node_B.name + ('.T' if trans_B else '')})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the matrix multiplication result of input values.

        Note
        ----
        For this homework, you can assume the matmul only works for 2d matrices.
        That being said, the test cases guarantee that input values are
        always 2d numpy.ndarray.
        """
        """TODO: Your code here"""
        assert len(input_values) == 2
        if input_values is None or node is None:
            raise ValueError('input_values or node is None')
        if len(input_values) != 2:
            raise ValueError('need 2 matrices')

        A, B = input_values
        trans_A = node.__getattr__('trans_A')
        trans_B = node.__getattr__('trans_B')
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T

        if A.shape[1] != B.shape[0]:
            raise ValueError('incompatible matrix sizes')
        result = np.matmul(A, B)
        return result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input.

        Note
        ----
        - Same as the `compute` method, you can assume that the input are 2d matrices.
        However, it would be a good exercise to think about how to handle
        more general cases, i.e., when input can be either 1d vectors,
        2d matrices, or multi-dim tensors.
        - You may want to look up some materials for the gradients of matmul.
        """
        """TODO: Your code here"""
        A, B = node.inputs
        trans_A = node.__getattr__('trans_A')
        trans_B = node.__getattr__('trans_B')

        if not trans_A and not trans_B:
            grad_A = self.__call__(output_grad, B, False, True)
            grad_B = self.__call__(A, output_grad, True, False)

        if trans_A and trans_B:
            grad_A = self.__call__(B, output_grad, True, True)
            grad_B = self.__call__(output_grad, A, True, True)

        if trans_A and not trans_B:
            grad_A = self.__call__(B, output_grad, False, True)
            grad_B = self.__call__(A, output_grad, False, True)

        if not trans_A and trans_B:
            grad_A = self.__call__(output_grad, B, False, False)
            grad_B = self.__call__(output_grad, A, True, False)

        return [grad_A, grad_B]


class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.zeros(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.ones(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()


class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def get_topological_sort(self, input_values: Dict[Node, np.ndarray], curr_eval_node: Node) -> List[Node]:
        if not curr_eval_node.inputs:
            return [curr_eval_node]

        curr_topo_list = []
        curr_eval_node_inputs = curr_eval_node.inputs
        for node_input in curr_eval_node_inputs:
            # check duplicates
            recur = self.get_topological_sort(input_values, node_input)
            for node in recur:
                if node not in curr_topo_list:
                    curr_topo_list.append(node)
            #curr_topo_list = curr_topo_list + self.get_topological_sort(input_values, node_input)
        if curr_eval_node not in curr_topo_list:
            curr_topo_list.append(curr_eval_node)

        return curr_topo_list

    def run(self, input_values: Dict[Node, np.ndarray]) -> List[np.ndarray]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, np.ndarray]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[np.ndarray]
            The list of values for nodes in `eval_nodes` field.
        """
        """TODO: Your code here"""

        for key, val in input_values.items():
            if not val.size > 0:
                raise ValueError('input node value not given')

        for node in self.eval_nodes:
            print(node)

        results_all_eval_nodes = []
        for i in range(len(self.eval_nodes)):
            curr_eval_nodes = self.eval_nodes[i]
            topological_sort = self.get_topological_sort(input_values, curr_eval_nodes)

            for curr_node in topological_sort:
                if curr_node.inputs:
                    curr_node_input_vals = []

                    for curr_node_inp in curr_node.inputs:
                        for key, val in input_values.items():
                            if str(curr_node_inp) == str(key):
                                if type(val) == list:
                                    curr_node_input_vals = curr_node_input_vals + val
                                else:
                                    curr_node_input_vals.append(val)
                                break

                    curr_res = curr_node.op.compute(curr_node, curr_node_input_vals)
                    input_values[curr_node] = curr_res

            results_all_eval_nodes.append(input_values[curr_eval_nodes])

        return results_all_eval_nodes


def get_topo_sort_grad(output_node: Node) -> List[Node]:
    if not output_node.inputs:
        return [output_node]

    curr_topo_list = []
    for node_input in output_node.inputs:
        # check dup
        recur = get_topo_sort_grad(node_input)
        for node in recur:
            if node not in curr_topo_list:
                curr_topo_list.append(node)
    if output_node not in curr_topo_list:
        curr_topo_list.append(output_node)

    return curr_topo_list

def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """

    """TODO: Your code here"""
    topological_sort = get_topo_sort_grad(output_node)
    topological_sort.reverse()
 
    node_to_grad = {output_node: [ones_like(output_node)]}
    for node in topological_sort:
        grads = node_to_grad[node]
        v_i = grads[0]
        if len(grads) > 0:
            for v_ij in grads[1:]:
                v_i = v_i + v_ij

        if node.inputs:
            v_ki = node.op.gradient(node, v_i)
            for i in range(len(node.inputs)):
                inp_node = node.inputs[i]
                if inp_node not in node_to_grad:
                    node_to_grad[inp_node] = []
                node_to_grad[inp_node].append(v_ki[i])

    result = []
    for node in nodes:
        grads = node_to_grad[node]
        v_i = grads[0]
        if len(grads) > 0:
            for v_ij in grads[1:]:
                v_i = v_i + v_ij
        result.append(v_i)

    return result
