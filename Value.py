class Value:

    def __init__(self, x):
        self.value = x
        self.grad = 0
        self.inputs = []
        self.input_op = None

    def __add__(self, other):
        s = Value(self.value + other.value)
        s.inputs = [self, other]
        s.input_op = Value.__add__
        return s
        
    def __mul__(self, other):
        prod = Value(self.value * other.value)
        prod.inputs = [self, other]
        prod.input_op = Value.__mul__
        return prod

    @staticmethod
    def _backward_add(grad, inputs):
        assert len(inputs) == 2, '__add__ must take 2 operands.'
        return [grad, grad]

    @staticmethod
    def _backward_mul(grad, inputs):
        assert len(inputs) == 2, '__mul__ must take 2 operands.'
        return [grad * inputs[1].value, grad * inputs[0].value]

    @classmethod
    def _backward_func(cls, func):
        if func == Value.__add__:
            return cls._backward_add
        if func == Value.__mul__:
            return cls._backward_mul
        assert False, f'{func} not supported.'

    def backward(self, grad=1):
        self.grad = grad
        if self.input_op is None:
            return None
        backward_func = Value._backward_func(self.input_op)
        input_grads = backward_func(grad, self.inputs)
        for input, input_grad in zip(self.inputs, input_grads):
            input.backward(input_grad)
