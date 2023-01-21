class Value:

    def __init__(self, x):
        self.value = x
        self.grad = 0
        self.inputs = []
        self.input_op = None

    def __add__(self, other):
        out = Value(self.value + other.value)
        out.inputs = [self, other]
        out.input_op = Value.__add__
        return out
        
    def __mul__(self, other):
        out = Value(self.value * other.value)
        out.inputs = [self, other]
        out.input_op = Value.__mul__
        return out

    def relu(self):
        out = Value(self.value * int(self.value > 0))
        out.inputs = [self]
        out.input_op = Value.relu
        return out
    
    @staticmethod
    def _backward_add(grad, inputs):
        assert len(inputs) == 2, '__add__ must take 2 operands.'
        return [grad, grad]

    @staticmethod
    def _backward_mul(grad, inputs):
        assert len(inputs) == 2, '__mul__ must take 2 operands.'
        return [grad * inputs[1].value, grad * inputs[0].value]

    @staticmethod
    def _backward_relu(grad, inputs):
        assert len(inputs) == 1, 'relu must take 1 operand.'
        return [grad * int(inputs[0].value > 0)]

    @classmethod
    def _backward_func(cls, func):
        if func == Value.__add__:
            return cls._backward_add
        if func == Value.__mul__:
            return cls._backward_mul
        if func == Value.relu:
            return cls._backward_relu
        assert False, f'{func} not supported.'

    def backward(self, grad=1):
        self.grad = grad
        if self.input_op is None:
            return None
        backward_func = Value._backward_func(self.input_op)
        input_grads = backward_func(grad, self.inputs)
        for input, input_grad in zip(self.inputs, input_grads):
            input.backward(input_grad)
