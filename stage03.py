import numpy as np
from dezero import Variable


def rosenbrock(x0: Variable, x1: Variable) -> Variable:
    return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2


def main():
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))

    lr = 0.001  # Learning Rate
    iters = 1000

    for i in range(iters):
        print(f"{x0=}, {x1=}")
        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()

        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad


if __name__ == "__main__":
    main()
