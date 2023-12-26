import math

import numpy as np

import dezero.datasets
from dezero import Variable, Model, DataLoader
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP


#
# class TwoLayerNet(Model):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.l1 = L.Linear(hidden_size)
#         self.l2 = L.Linear(out_size)
#
#     def forward(self, x):
#         y = self.l1(x)
#         y = F.sigmoid(y)
#         y = self.l2(y)
#         return y
#
#
# x = Variable(np.random.randn(5, 10), name="x")
# model = TwoLayerNet(100, 10)
# model.plot(x)
#
# # 層をモデルとしてまとめて定義 出力サイズのみ指定
# model = L.Layer()
# model.l1 = L.Linear(5)
# model.l2 = L.Linear(1)
#
#
# def predict(x):
#     y = model.l1(x)
#     y = F.sigmoid(y)
#     y = model.l2(y)
#     return y
#
#
# def main():
#     np.random.seed(0)
#     x = np.random.rand(100, 1)
#     y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
#
#     lr = 0.2
#     iters = 10000
#     hidden_size = 10
#
#     model = MLP((hidden_size, 1))
#     optimizer = optimizers.SGD(lr)
#     optimizer.setup(model)
#
#     for i in range(iters):
#         y_pred = model(x)
#         loss = F.mean_squared_error(y, y_pred)
#
#         model.cleargrads()
#         loss.backward()
#         optimizer.update()
#
#         if i % 1000 == 0:
#             print(loss)


def step49():
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    train_set = dezero.datasets.Spiral()
    model = MLP((hidden_size, 10))
    optimizer = optimizers.SGD(lr).setup(model)

    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)

    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)
        sum_loss = 0.0

        for i in range(max_iter):
            batch_index = index[i * batch_size: (i + 1) * batch_size]
            batch = [train_set[i] for i in batch_index]
            batch_x = np.array([b[0] for b in batch])
            batch_t = np.array([b[1] for b in batch])

            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        avg_loss = sum_loss / data_size
        print(f"epoch: {epoch + 1}, loss: {avg_loss:.2f}")


def step50():
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    train_set = dezero.datasets.Spiral(train=True)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    test_set = dezero.datasets.Spiral(train=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, 10))
    optimizer = optimizers.SGD(lr).setup(model)

    for epoch in range(max_epoch):
        sum_loss = 0.0
        sum_acc = 0.0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        print(f"\n\nepoch: {epoch + 1}")
        print(f"train loss: {sum_loss / len(train_set):.4f}")
        print(f"train accuracy: {sum_acc / len(train_set):.4f}")

        sum_loss = 0.0
        sum_acc = 0.0

        with dezero.no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)

                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

            print(f"test loss: {sum_loss / len(test_set):.4f}")
            print(f"test accuracy: {sum_acc / len(test_set):.4f}")


def step51():
    max_epoch = 5
    batch_size = 100
    hidden_size = 1000

    train_set = dezero.datasets.MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    test_set = dezero.datasets.MNIST(train=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    # 86% -層を増やす-> 67% -ReLU-> 93% -Adam-> 98%
    model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
    optimizer = optimizers.Adam().setup(model)

    for epoch in range(max_epoch):
        sum_loss = 0.0
        sum_acc = 0.0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        print(f"\n\nepoch: {epoch + 1}")
        print(f"train loss: {sum_loss / len(train_set):.4f}")
        print(f"train accuracy: {sum_acc / len(train_set):.4f}")

        sum_loss = 0.0
        sum_acc = 0.0

        with dezero.no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)

                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

            print(f"test loss: {sum_loss / len(test_set):.4f}")
            print(f"test accuracy: {sum_acc / len(test_set):.4f}")


if __name__ == "__main__":
    step51()
