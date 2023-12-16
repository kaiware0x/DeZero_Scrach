import numpy as np

from dezero import Variable

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y: Variable = x0 + x1
print(f"{y=}")

y.backward()
print(f"{x1.grad=}")
