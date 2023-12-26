import numpy as np
import math

import dezero.core
from dezero import Variable, Function, as_variable, as_array, utils, cuda


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
        xp = cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gys):
        x = self.inputs[0]
        x_grad = exp(x) * gys  # 上流からの微分量を受け継いでいく.(連鎖律)
        return x_grad


def exp(x):
    return Exp()(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.log(x)

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Sin(Function):
    def forward(self, x: np.ndarray):
        xp = cuda.get_array_module(x)
        return xp.sin(x)

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
        xp = cuda.get_array_module(x)
        return xp.cos(x)

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
        xp = cuda.get_array_module(x)
        return xp.tanh(x)

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
        xp = cuda.get_array_module(x)
        return xp.transpose(x)

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
        xp = cuda.get_array_module(x)
        self.input_shape = x.shape
        return xp.broadcast_to(x, self.output_shape)

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


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None  # メモリ効率のため余計なデータは削除
    return y


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


def sigmoid_simple(x):
    x = as_variable(x)
    return 1 / (1 + exp(-x))


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        xp.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis=1):
        """

        :param axis: デフォルト1にして各行ごとの和を計算する.
        """
        self.axis = axis

    def forward(self, x: np.ndarray):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        xp = cuda.get_array_module(x)
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[xp.arange(N), t.ravel()]
        y = -log_p.sum() / xp.float32(N)
        return y

    def backward(self, gy):
        xp = cuda.get_array_module(gy)

        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        # convert to one-hot
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class ReLU(Function):
    def forward(self, x: np.ndarray):
        xp = cuda.get_array_module(x)
        return xp.maximum(x, 0.0)

    def backward(self, gy: Variable):
        xp = cuda.get_array_module(gy)
        x, = self.inputs
        mask: xp.ndarray = x.data > 0  # 0より大きい要素がTrueのbool型のndarrayができる
        gx = gy * mask  # Mul()
        return gx


def relu(x):
    return ReLU()(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if dezero.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


# =============================================================================
# conv2d / col2im / im2col / basic_math
# =============================================================================
from dezero.functions_conv import conv2d
from dezero.functions_conv import deconv2d
from dezero.functions_conv import conv2d_simple
from dezero.functions_conv import im2col
from dezero.functions_conv import col2im
from dezero.functions_conv import pooling_simple
from dezero.functions_conv import pooling
from dezero.functions_conv import average_pooling
from dezero.core import add
from dezero.core import sub
from dezero.core import rsub
from dezero.core import mul
from dezero.core import div
from dezero.core import neg
from dezero.core import pow
