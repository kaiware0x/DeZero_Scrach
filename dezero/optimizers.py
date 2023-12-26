import math

from dezero import cuda
from dezero.core import Parameter


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        """

        :param target: Layer|Model
        :return:
        """
        self.target = target
        return self

    def update(self):
        # None以外のパラメータを取得
        params = [p for p in self.target.params() if p.grad is not None]

        # 前処理
        for f in self.hooks:
            f(params)

        # パラメータの更新
        for param in params:
            self.update_one(param)

    def update_one(self, param: Parameter):
        """

        :param param: Parameter
        :return:
        """
        raise NotImplementedError

    def add_hook(self, f):
        self.hooks.append(f)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    確率的勾配降下法
    """

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param: Parameter):
        param.data -= self.lr * param.grad.data


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param: Parameter):
        xp = cuda.get_array_module(param.data)

        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]  # 前回の速度を取得
        v *= self.momentum  # 若干ブレーキを掛ける
        v -= self.lr * param.grad.data
        param.data += v


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + eps)