import numpy as np
import unittest

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

    def set_creator(self, func) -> None:
        """
        このVariableを生成したのは誰かを保持する.
        :param func:
        :return: None
        """
        self.creator = func

    def backward(self):
        # 勾配の初期値を1にする(dy/dy)
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]  # 自身の生みの親関数リスト. 再起よりもループで処理した方が処理時間も早く、後々の実装も楽になる
        while len(funcs) != 0:
            f: Function = funcs.pop()  # 末尾からpop
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)  # dy/dx 自身の微分量から入力微分量を計算

            if x.creator is not None:  # Noneの場合は生みの親関数がいない→ユーザ入力したVariable
                funcs.append(x.creator)


class Function:
    """
    すべての関数の基底クラス
    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)  # 出力変数に生みの親を覚えさせる.
        self.input = input  # 逆伝播で使うので覚えておく.
        self.output = output
        return output

    def forward(self, x: np.ndarray):
        """
        順伝播を行う.
        :param x:
        :return:
        """
        raise NotImplementedError()

    def backward(self, grad_y: np.ndarray):
        """
        誤差逆伝播を行う.
        :param grad_y: forward出力側微分量
        :return: forward入力側微分用
        """
        raise NotImplementedError()


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class Square(Function):
    def forward(self, x: np.ndarray):
        return x ** 2

    def backward(self, grad_y: np.ndarray):
        x = self.input.data
        grad_x = 2 * x * grad_y
        return grad_x


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
        expected = np.array(6.0) # 2x = 2*3 = 6
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        """勾配確認"""
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        x_grad_num = numerical_diff(square, x)
        ok = np.allclose(x.grad, x_grad_num) # だいたい同じか判定する.
        self.assertTrue(ok)

class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)

    def backward(self, grad_y: np.ndarray):
        x = self.input.data
        grad_x = np.exp(x) * grad_y  # 上流からの微分量を受け継いでいく.
        return grad_x


def exp(x):
    return Exp()(x)




def main():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()  # 誤差逆伝播
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
