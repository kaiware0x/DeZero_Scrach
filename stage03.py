import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph


def main():
    x = Variable(np.random.randn(1, 2, 3))
    print(f"{x=}")
    y = x.reshape((2, 3))
    print(f"{y=}")
    y = x.reshape(2, 3)
    print(f"{y=}")

    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = "x"
    y.name = "y"
    y.backward(create_graph=True)
    #
    # iters = 4
    #
    # for i in range(iters):
    #     gx = x.grad
    #     x.cleargrad()
    #     gx.backward(create_graph=True)
    #
    # gx = x.grad
    # gx.name = "gx" + str(iters+1)
    # plot_dot_graph(gx, verbose=False, to_file="tanh.png")

    #
    # ###
    # ###
    # ###
    # x = Variable(np.linspace(-7, 7, 200))
    # y = F.sin(x)
    # y.backward(create_graph=True)
    #
    # logs = [y.data.flatten()]
    #
    # for i in range(3):
    #     logs.append(x.grad.data.flatten())
    #     gx = x.grad
    #     x.cleargrad()  # 1階微分の結果と2階微分の結果が加算されないようにReset
    #     gx.backward(create_graph=True)
    #
    # labels = ["y=sin(x)", "y'", "y''", "y'''"]
    # for i, v in enumerate(logs):
    #     plt.plot(x.data, logs[i], label=labels[i])
    # plt.legend(loc="lower right")
    # plt.show()

    #
    # # 2階微分によるニュートン法
    # x = Variable(np.array(2.0))
    # iters = 10
    #
    # for i in range(iters):
    #     print(i, x)
    #     y = f(x)
    #     x.cleargrad()
    #     y.backward(create_graph=True)
    #
    #     gx = x.grad
    #     x.cleargrad()
    #     gx.backward()
    #     gx2 = x.grad
    #
    #     x.data -= gx.data / gx2.data


if __name__ == "__main__":
    main()
