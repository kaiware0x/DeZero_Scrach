import numpy as np
import unittest
import weakref

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    else:
        return x


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.data: np.ndarray = data
        self.grad: np.ndarray = None
        self.creator = None
        self.generation = 0  # selfVariableインスタンスは計算グラフの何世代目か？

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
        同じ変数を使って再度別のforward計算を行うと勾配計算が誤って加算されてしまうため、リセットする仕組みを用意する.
        :return:
        """
        self.grad = None

    def backward(self):
        # 勾配の初期値を1にする(dy/dy)
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 自身の生みの親関数リスト. 再起よりもループで処理した方が処理時間も早く、後々の実装も楽になる
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f) # 関数リストに追加
                seen_set.add(f) # 追加済みセットに追加. 関数を二重で取得することを防ぐ.
                funcs.sort(key=lambda x : x.generation) # 世代昇順にソート
        add_func(self.creator)

        while len(funcs) != 0:
            f: Function = funcs.pop()  # 末尾からpop→最も新しい世代の関数を取得
            gys = [output().grad for output in f.outputs] # 弱参照から値を取得
            gxs = f.backward(*gys)  # アンパッキングして渡す.
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    # xの勾配がすでに計算済みの場合は上書きせず加算する必要がある.
                    # (入力に同じ変数が複数使われていた場合)
                    x.grad = x.grad + gx  # +=で書くと後々問題になる

                if x.creator is not None:  # Noneの場合は生みの親関数がいない→ユーザ入力したVariable
                    add_func(x.creator)


class Function:
    """
    すべての関数の基底クラス
    """

    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # アンパッキングして渡す.
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        # 入力変数のうち最も世代が新しいものをselfの世代とする.
        self.generation = max([x.generation for x in inputs])

        for output in outputs:
            output.set_creator(self)  # 出力変数に生みの親を覚えさせる.

        self.inputs = inputs  # 逆伝播で使うので覚えておく.
        self.outputs = [weakref.ref(output) for output in outputs]

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def forward(self, xs: tuple[np.ndarray]) -> tuple[np.ndarray]:
        """
        順伝播を行う.
        :param xs:
        :return:
        """
        raise NotImplementedError()

    def backward(self, y_grads: tuple[np.ndarray]) -> tuple[np.ndarray]:
        """
        誤差逆伝播を行う.
        :param y_grads: forward出力側の微分量
        :return: forward入力側の微分量
        """
        raise NotImplementedError()


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, y_grads):
        x = self.inputs[0].data
        x_grad = 2 * x * y_grads  # 微分の連鎖律
        return x_grad


def square(x):
    return Square()(x)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)  # 2x = 2*3 = 6
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        """勾配確認"""
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        x_grad_num = numerical_diff(square, x)
        ok = np.allclose(x.grad, x_grad_num)  # だいたい同じか判定する.
        self.assertTrue(ok)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, y_grads):
        x = self.inputs[0].data
        x_grad = np.exp(x) * y_grads  # 上流からの微分量を受け継いでいく.(連鎖律)
        return x_grad


def exp(x):
    return Exp()(x)


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, y_grads):
        return y_grads, y_grads


def add(x0, x1):
    return Add()(x0, x1)


def main():
    x = Variable(np.array(2))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()
    print(f"{y.data=}")
    print(f"{x.grad=}")

    y = Variable(np.array(3))
    z = add(square(x), square(y))
    z.backward()
    print(f"{z.data=}")
    print(f"{x.grad=}")
    print(f"{y.grad=}")
    #
    # x = Variable(np.array(2))
    # y = Variable(np.array(3))
    # z = add(square(x), square(y))
    x.cleargrad()
    y.cleargrad()
    z.cleargrad()
    z.backward()
    print(f"{z.data=}")
    print(f"{x.grad=}")
    print(f"{y.grad=}")

    # x = [Variable(np.array(0.5))]
    x = Variable(np.array(0.5))
    z = square(exp(square(x)))
    z.backward()  # 誤差逆伝播
    print(f"誤差逆伝播法 自動微分: {x.grad}")


if __name__ == "__main__":
    main()
    unittest.main()

#
# A = Square()
# B = Exp()
# C = Square()
#
# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad = B.backward(b.grad)
# x.grad = A.backward(a.grad)
# print(f"誤差逆伝播法: {x.grad}")
#
#
# def f(x):
#     F1 = Square()
#     F2 = Exp()
#     F3 = Square()
#     return F3(F2(F1(x)))
#
#
# x = Variable(np.array(0.5))
# dy = numerical_diff(f, x)
# print(f"数値微分: {dy}")
#
# f = Square()
# x = Variable(np.array(2.0))
# dy = numerical_diff(f, x)
# print(dy)
#
# sq_func1 = Square()
# exp_func = Exp()
# sq_func2 = Square()
#
# x = Variable(np.array(0.5))
# a = sq_func1(x)
# b = exp_func(a)
# y = sq_func2(b)
# print(type(y))
# print(y.data)
