import math

import numpy as np
import weakref
import contextlib

import dezero


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value):
    """
    with構文でConfigの値を制御できるようにする.
    :param name:
    :param value:
    :return:
    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    else:
        return x


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Variable:
    __array_priority__ = 200  # np.ndarrayのオペレータよりもoverloadが優先されるようにする.

    def __init__(self, data: np.ndarray, name: str = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.data: np.ndarray = data
        self.name: str = name
        self.grad: Variable = None
        self.creator = None
        self.generation = 0  # selfVariableインスタンスは計算グラフの何世代目か？

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)

    @property
    def T(self):
        return self.transpose()

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def set_creator(self, func) -> None:
        """
        このVariableを生成したのは誰かを保持する.
        :param func:
        :return: None
        """
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        """
        同じ変数を使って再度別のforward計算を行うと勾配計算が誤って加算されてしまうため、
        リセットする仕組みを用意する.
        :return:
        """
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        # 勾配の初期値を1にする(dy/dy)
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        # 自身の生みの親関数リスト. 再起よりもループで処理した方が処理時間も早く、後々の実装も楽になる
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)  # 関数リストに追加
                seen_set.add(f)  # 追加済みセットに追加. 関数を二重で取得することを防ぐ.
                funcs.sort(key=lambda x: x.generation)  # 世代昇順にソート

        add_func(self.creator)

        while len(funcs) != 0:
            f: Function = funcs.pop()  # 末尾からpop→最も新しい世代の関数を取得
            gys = [output().grad for output in f.outputs]  # 弱参照から値を取得

            # デフォルトで勾配を保持しないモードで計算する.
            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)  # Functionの__call__が呼ばれる。
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        # xの勾配がすでに計算済みの場合は上書きせず加算する必要がある.
                        # (入力に同じ変数が複数使われていた場合)
                        x.grad = x.grad + gx  # Add関数オブジェクトが呼ばれる.

                    # Noneの場合は生みの親関数がいない→ユーザ入力したVariable
                    if x.creator is not None:
                        add_func(x.creator)

            # メモリ使用量のため使用済みの勾配を破棄する。
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Function:
    """
    すべての関数の基底クラス
    """

    def __call__(self, *inputs):
        inputs_var: list[Variable] = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs_var]  # np.ndarrayを取得.
        ys = self.forward(*xs)  # アンパッキングして渡す.
        if not isinstance(ys, tuple):
            ys = (ys,)

        # yがスカラーの場合もあるのでas_arrayでnp.ndarrayにしてからVariableへ
        outputs = [Variable(as_array(y)) for y in ys]

        # 推論時は勾配計算はいらない
        # 学習中forward計算のときにはTrueになっている
        if Config.enable_backprop:
            # inputs --> self() --> outputs

            # 入力変数のうち最も世代が新しいものをselfの世代とする.
            self.generation = max([x.generation for x in inputs_var])

            for output in outputs:
                output.set_creator(self)  # 出力変数に生みの親を覚えさせる.

            self.inputs: list[Variable] = inputs_var  # 逆伝播で使うので覚えておく.
            self.outputs = [weakref.ref(output) for output in outputs]

        if len(outputs) > 1:
            return outputs
        else:
            return outputs[0]

    def forward(self, xs: tuple[np.ndarray]) -> tuple[np.ndarray]:
        """
        順伝播を行う.
        :param xs:
        :return:
        """
        raise NotImplementedError()

    def backward(self, gys: tuple[Variable]) -> tuple[Variable]:
        """
        誤差逆伝播を行う.
        :param gys: forward出力側の微分量
        :return: forward入力側の微分量
        """
        raise NotImplementedError()


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return x0 + x1

    def backward(self, gy: Variable):
        gx0, gx1 = gy, gy
        # forwardでBroadcastによる配列の拡張が行われていた場合は元のShapeに戻す
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Sub(Function):

    def forward(self, x0: np.ndarray, x1: np.ndarray):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return x0 - x1

    def backward(self, gy: Variable):
        gx0, gx1 = gy, -gy
        # forwardでBroadcastによる配列の拡張が行われていた場合は元のShapeに戻す
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Mul(Function):
    """
    乗算を行う関数オブジェクト
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return x0 * x1

    def backward(self, gy: Variable):
        x0, x1 = self.inputs[0], self.inputs[1]
        gx0 = x1 * gy
        gx1 = x0 * gy

        # forwardでBroadcastによる配列の拡張が行われていた場合は元のShapeに戻す
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)

        # それぞれ偏微分するのでx0とx1が入れ替わる
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class Div(Function):

    def forward(self, x0: np.ndarray, x1: np.ndarray):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return x0 / x1

    def backward(self, gy: Variable):
        x0, x1 = self.inputs[0], self.inputs[1]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)

        # forwardでBroadcastによる配列の拡張が行われていた場合は元のShapeに戻す
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)

        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0]
        return gy * self.c * x ** (self.c - 1)


def pow(x, c):
    return Pow(c)(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

class Neg(Function):

    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


def setup_variable():
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
