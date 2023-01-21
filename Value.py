import numpy as np

class Value:

    def __init__(self, x):
        self.value = np.array(x, dtype=np.float32)
        self.grad = np.zeros_like(x)
        self.inputs = []
        self.input_op = None
        self.cache = None

    def __add__(self, other):
        assert self.value.shape == other.value.shape
        out = Value(self.value + other.value)
        out.inputs = (self, other)
        out.input_op = Value.__add__
        return out
        
    def __mul__(self, other):
        assert self.value.shape == other.value.shape
        out = Value(self.value * other.value)
        out.inputs = (self, other)
        out.input_op = Value.__mul__
        return out

    def relu(self):
        mask = (self.value > 0).astype(np.float32)
        out = Value(self.value * mask)
        out.inputs = (self,)
        out.input_op = Value.relu
        out.cache = (mask,)
        return out

    def softmax(self, y):
        assert len(y.shape) == 1 and len(self.value.shape) == 2
        y = np.array(y, dtype=np.int32)
        assert self.value.shape[:-1] == y.shape
        exp = np.exp(self.value)
        norm = exp / np.sum(exp, axis=1)[:, np.newaxis]
        out = Value(np.mean(-np.log(norm[np.arange(y.shape[0]), y])))
        out.inputs = (self,)
        out.input_op = Value.softmax
        out.cache = (norm, y)
        return out

    @staticmethod
    def _backward_add(grad, inputs, cache):
        assert len(inputs) == 2, '__add__ must take 2 operands.'
        return [grad, grad]

    @staticmethod
    def _backward_mul(grad, inputs, cache):
        assert len(inputs) == 2, '__mul__ must take 2 operands.'
        return [grad * inputs[1].value, grad * inputs[0].value]

    @staticmethod
    def _backward_relu(grad, inputs, cache):
        assert len(inputs) == 1, 'relu must take 1 operand.'
        return [grad * cache[0]]

    @staticmethod
    def _backward_softmax(grad, inputs, cache):
        assert len(inputs) == 1, 'softmax only backprop wrt 1 operand.'
        norm, y = cache
        din = norm
        din[np.arange(y.shape[0]), y] -= 1.0
        din /= y.shape[0]
        return [grad * din]

    @classmethod
    def _backward_func(cls, func):
        if func == Value.__add__:
            return cls._backward_add
        if func == Value.__mul__:
            return cls._backward_mul
        if func == Value.relu:
            return cls._backward_relu
        if func == Value.softmax:
            return cls._backward_softmax
        assert False, f'{func} not supported.'

    def zero_grad(self):
        self.grad = 0.0
        for input in self.inputs:
            input.zero_grad()

    def backward(self, grad=1.0):
        self.grad += grad
        if self.input_op is None:
            return None
        backward_func = Value._backward_func(self.input_op)
        input_grads = backward_func(grad, self.inputs, self.cache)
        for input, input_grad in zip(self.inputs, input_grads):
            input.backward(input_grad)
