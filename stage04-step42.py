import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable
import dezero.functions as F

np.random.seed(0)

x = np.random.rand(100, 1)
y = 2 * x + 5 + np.random.rand(100, 1)
xmin = x.min(axis=0)
xmax = x.max(axis=0)

x = Variable(x)
y = Variable(y)

W = Variable(np.zeros([1, 1]))
b = Variable(np.zeros(1))


def predict(x):
    return F.matmul(x, W) + b


def mse(x0, x1) -> Variable:
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 1000

for i in range(iters):
    y_pred = predict(x)
    loss = mse(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    print(f"{i=}, {W=}, {b=}, {loss=}")

y_pred = W * np.concatenate([xmin, xmax], axis=0) + b
y_pred = y_pred.data[0]

plt.figure(figsize=(8, 6))
plt.scatter(x.data, y.data, color='blue', label='Data (y=2x+5)')
plt.plot([xmin, xmax], y_pred, color="red", label=f"Predict (y={W.data[0][0]:.3}x+{b.data[0]:.3})")
plt.legend()
plt.show()
