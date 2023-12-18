import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable
import dezero.functions as F

np.random.seed(0)

x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
xmin = x.min(axis=0)
xmax = x.max(axis=0)

x = Variable(x)
y = Variable(y)

# 重みの初期化
I, H, O = 1, 10, 1 # Input, Hidden, Output
W1 = Variable(0.01 *  np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 *  np.random.randn(H, O))
b2 = Variable(np.zeros(O))

def predict(x):
    y = F.linear_simple(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = F.linear_simple(y, W2, b2)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()

    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:
        print(f"{i=}, {loss=}")


y_pred = predict(x)

plt.figure(figsize=(8, 6))
plt.scatter(x.data, y.data, color='blue', label='Data (y=sin(x))')
plt.scatter(x.data, y_pred.data, color="red", label=f"Predict")
plt.legend()
plt.show()
