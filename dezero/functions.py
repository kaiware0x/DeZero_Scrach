import numpy as np
import math

from dezero import Variable
from dezero.core import Function
from dezero.core import as_array
from dezero.core import as_variable
from dezero import utils


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gys):
        x: Variable = self.inputs[0]
        x_grad = 2 * x * gys  # 微分の連鎖律
        return x_grad


def square(x):
    return Square()(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gys):
        x = self.inputs[0]
        x_grad = exp(x) * gys  # 上流からの微分量を受け継いでいく.(連鎖律)
        return x_grad


def exp(x):
    return Exp()(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Sin(Function):
    def forward(self, x: np.ndarray):
        return np.sin(x)

    def backward(self, gy: Variable):
        x: Variable = self.inputs[0]
        return gy * cos(x)


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001):
    """
    マクローリン展開を用いてSinを計算する.
    :param x:
    :param threshold:
    :return:
    """
    y = 0
    for i in range(100_000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Cos(Function):
    def forward(self, x: np.ndarray):
        return np.cos(x)

    def backward(self, gy: Variable):
        x = self.inputs[0]
        return - gy * sin(x)


def cos(x):
    return Cos()(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Tanh(Function):
    def forward(self, x: np.ndarray):
        return np.tanh(x)

    def backward(self, gy: Variable):
        y = self.outputs[0]()  # to strong ref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Reshape(Function):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, x: np.ndarray):
        self.input_shape = x.shape
        return x.reshape(self.output_shape)

    def backward(self, gy: Variable):
        return reshape(gy, self.input_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Transpose(Function):
    def forward(self, x: np.ndarray):
        return np.transpose(x)

    def backward(self, gy: Variable):
        return transpose(gy)


def transpose(x: Variable):
    return Transpose()(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray):
        self.input_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy: Variable):
        gy = utils.reshape_sum_backward(gy, self.input_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.input_shape)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class BroadcastTo(Function):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, x: np.ndarray):
        self.input_shape = x.shape
        return np.broadcast_to(x, self.output_shape)

    def backward(self, gy: Variable):
        return sum_to(gy, self.input_shape)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class SumTo(Function):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, x: np.ndarray):
        self.input_shape = x.shape
        return utils.sum_to(x, self.output_shape)

    def backward(self, gy: Variable):
        return broadcast_to(gy, self.input_shape)


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray):
        return x.dot(W)

    def backward(self, gy):
        x = self.inputs[0]
        W = self.inputs[1]
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class MeanSquaredError(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        diff = x0 - x1
        return (diff ** 2).sum() / len(diff)

    def backward(self, gy: Variable):
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        diff: Variable = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def mean_squared_error_simple(x0, x1) -> Variable:
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

def linear_simple(x, W, b=None):
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None # メモリ効率のため余計なデータは削除
    return y


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

def sigmoid_simple(x):
    x = as_variable(x)
    return 1 / (1 + exp(-x))
