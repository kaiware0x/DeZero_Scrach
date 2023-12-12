import unittest
from builtins import zip

import numpy as np

import dezero
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = dezero.square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = dezero.square(x)
        y.backward()
        expected = np.array(6.0)  # 2x = 2*3 = 6
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        """勾配確認"""
        x = Variable(np.random.rand(1))
        y = dezero.square(x)
        y.backward()
        x_grad_num = numerical_diff(dezero.square, x)
        ok = np.allclose(x.grad, x_grad_num)  # だいたい同じか判定する.
        self.assertTrue(ok)


class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(-1))
        y = dezero.exp(x)
        expected = np.e ** (-1)
        self.assertEqual(y.data, expected)


class AddTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 + x1
        expected = np.array(5.0)
        self.assertEqual(y.data, expected)


class SubTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 - x1
        expected = np.array(-1.0)
        self.assertEqual(y.data, expected)


class MulTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 * x1
        expected = np.array(6.0)
        self.assertEqual(y.data, expected)


class DivTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 / x1
        expected = np.array(2.0 / 3.0)
        self.assertEqual(y.data, expected)


class PowTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = x ** 3
        expected = np.array(8.0)
        self.assertEqual(y.data, expected)


class NegTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = -x
        expected = np.array(-2.0)
        self.assertEqual(y.data, expected)


class SphereTest(unittest.TestCase):
    def sphere(self, x, y):
        z = x ** 2 + y ** 2
        return z

    def test_sphere(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z: Variable = self.sphere(x, y)
        z.backward()
        self.assertEqual(x.grad, np.array(2.0))
        self.assertEqual(y.grad, np.array(2.0))


class MatyasTest(unittest.TestCase):
    def matyas(self, x, y):
        z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
        return z

    def test_matyas(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z: Variable = self.matyas(x, y)
        z.backward()
        self.assertAlmostEqual(x.grad, np.array(0.04))
        self.assertAlmostEqual(y.grad, np.array(0.04))


class GoldsteinPriceTest(unittest.TestCase):
    def goldstein(self, x, y):
        z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
            (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
        return z

    def test_goldstein(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z: Variable = self.goldstein(x, y)
        z.backward()
        self.assertEqual(x.grad, np.array(-5376.0))
        self.assertEqual(y.grad, np.array(8064.0))

        x.name = "x"
        y.name = "y"
        z.name = "z"
        plot_dot_graph(z, verbose=False, to_file="goldstein.png")


if __name__ == "__main__":
    unittest.main()
