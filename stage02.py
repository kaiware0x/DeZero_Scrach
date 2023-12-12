"""

"""

# 現在のスクリプトの親フォルダをモジュール探索パスに含め,
# dezeroフォルダが常に見つかるようにする.
if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable


def main():
    a = Variable(np.array(1.0))
    y: Variable = (a + 3) ** 2
    y.backward()
    print(f"{y=}")
    print(f"{a.grad=}")

    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    # y = add(mul(a, b), c)
    # y = a * b + c
    y = 3.0 * b + 1.0

    y.backward()

    print(f"{y=}")
    print(f"{a.grad=}")
    print(f"{b.grad=}")


if __name__ == "__main__":
    main()
