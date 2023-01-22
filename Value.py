import numpy as np

class Value:

    def __init__(self, x):
        self.value = np.array(x, dtype=np.float32)
        self.grad = np.zeros_like(self.value)
        self.inputs = []
        self.input_op = None
        self.cache = None

    def __add__(self, other):
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

    def mm(self, other):
        assert len(self.value.shape) <= 2 and len(other.value.shape) <= 2
        assert self.value.shape[-1] == other.value.shape[0]
        assert len(self.value.shape) > 1 or len(other.value.shape) > 1
        out = Value(self.value @ other.value)
        out.inputs = (self, other)
        out.input_op = Value.mm
        return out

    @property
    def T(self):
        out = Value(self.value.T)
        out.inputs = (self,)
        out.input_op = Value.T
        return out

    def relu(self):
        mask = (self.value > 0).astype(np.float32)
        out = Value(self.value * mask)
        out.inputs = (self,)
        out.input_op = Value.relu
        out.cache = (mask,)
        return out

    def cross_entropy_loss(self, y):
        y = np.array(y, dtype=np.uint32)
        exp = np.exp(self.value)
        norm = exp / np.sum(exp, axis=-1, keepdims=True)
        out = Value(np.mean(-np.log(norm[np.indices(y.shape), y])))
        out.inputs = (self,)
        out.input_op = Value.cross_entropy_loss
        out.cache = (norm, y)
        return out

    @staticmethod
    def _backward_add(grad, inputs, cache):
        assert len(inputs) == 2, '__add__ must take 2 operands.'
        vals = [input.value for input in inputs]
        t = lambda val: np.sum(grad.reshape(-1, *val.shape), axis=0)
        return [t(val) if val.shape != grad.shape else grad for val in vals]

    @staticmethod
    def _backward_mul(grad, inputs, cache):
        assert len(inputs) == 2, '__mul__ must take 2 operands.'
        return [grad * inputs[1].value, grad * inputs[0].value]

    @staticmethod
    def _backward_relu(grad, inputs, cache):
        assert len(inputs) == 1, 'relu must take 1 operand.'
        return [grad * cache[0]]

    @staticmethod
    def _backward_cross_entropy_loss(grad, inputs, cache):
        assert len(inputs) == 1, 'cross_entropy_loss only backprops wrt 1 operand.'
        norm, y = cache
        din = norm
        din[np.indices(y.shape), y] -= 1.0
        din /= np.prod(y.shape)
        return [grad * din]

    @staticmethod
    def _backward_mm(grad, inputs, cache):
        assert len(inputs) == 2, 'mm must take 2 operands.'
        x, y = inputs[0].value, inputs[1].value
        y_ = np.expand_dims(y, axis=1) if len(y.shape) == 1 else y
        grad_ = np.expand_dims(grad, axis=1) if len(grad.shape) == 1 else grad
        dx = np.reshape(grad_ @ y_.T, x.shape)
        dy = np.reshape(x.T @ grad_, y.shape)
        return [dx, dy]

    @staticmethod
    def _backward_T(grad, inputs, cache):
        assert len(inputs) == 1, 'T must take 1 operand.'
        return [grad.T]

    @classmethod
    def _backward_func(cls, func):
        if func == Value.__add__:
            return cls._backward_add
        if func == Value.__mul__:
            return cls._backward_mul
        if func == Value.relu:
            return cls._backward_relu
        if func == Value.cross_entropy_loss:
            return cls._backward_cross_entropy_loss
        if func == Value.mm:
            return cls._backward_mm
        if func == Value.T:
            return cls._backward_T
        assert False, f'{func} not supported.'

    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)
        for input in self.inputs:
            input.zero_grad()

    def backward(self, grad=1.0):
        self.grad += grad
        if self.input_op is None:
            return None
        backward_func = Value._backward_func(self.input_op)
        input_grads = backward_func(self.grad, self.inputs, self.cache)
        for input, input_grad in zip(self.inputs, input_grads):
            if type(input_grad) == int or type(input_grad) == float:
                input_grad = np.array(input_grad, dtype=input.dtype)
            input.backward(input_grad)
