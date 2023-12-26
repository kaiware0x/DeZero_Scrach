import time
import numpy as np
import matplotlib.pyplot as plt

import dezero.datasets
import dezero.functions as F
import dezero.layers as L
from dezero import DataLoader, optimizers, Model, SeqDataLoader
from dezero.models import MLP


def step52():
    max_epoch = 5
    batch_size = 100

    train_set = dezero.datasets.MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)
    model = MLP((1000, 10))
    optimizer = optimizers.SGD().setup(model)

    # if dezero.cuda.gpu_enable:
    #     train_loader.to_gpu()
    #     model.to_gpu()

    for epoch in range(max_epoch):
        start = time.time()
        sum_loss = 0.0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(t)

        elapsed_time = time.time() - start
        print(f"\n\nepoch: {epoch + 1}, loss: {sum_loss / len(train_set):.4f}, time: {elapsed_time:.4f}")

    return


def step57():
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
    # kernelが動けるのは入力データ7x7のうち3x3の領域 -> col1.shape[0] = 3x3 = 9
    # 1つのkernelが持つパラメータ数 = in_ch * ksize_h * ksize_w = 3x5x5 = 75
    print(col1.shape)  # (9, 75)

    x2 = np.random.rand(10, 3, 7, 7)
    kernel_size = 5, 5
    stride = 1, 1
    pad = 0, 0
    col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
    print(col2.shape)  # (90, 75) batch_size=10なので単純に前回の10倍サイズ
    return


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


def step59():
    max_epoch = 100
    hidden_size = 100
    bptt_length = 30  # Backprop Through Time
    train_set = dezero.datasets.SinCurve(train=True)
    seq_len = len(train_set)

    model = SimpleRNN(hidden_size, 1)
    optim = dezero.optimizers.Adam().setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0

        for x, t in train_set:
            x = x.reshape(1, 1)
            y = model(x)
            loss += F.mean_squared_error(y, t)
            count += 1

            if count % bptt_length == 0 or count == seq_len:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optim.update()
        avg_loss = float(loss.data) / count
        print(f"| epoch {epoch} | loss {avg_loss}")

    # test

    xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
    model.reset_state()
    pred_list = []

    with dezero.no_grad():
        for x in xs:
            x = np.array(x).reshape(1, 1)
            y = model(x)
            pred_list.append(float(y.data.flatten()))

    plt.plot(np.arange(len(xs)), xs, label="y=cos(x)")
    plt.plot(np.arange(len(xs)), pred_list, label="predict")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return


class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y


def step60():
    max_epoch = 100
    batch_size = 30
    hidden_size = 100
    bptt_length = 30  # Backprop Through Time
    train_set = dezero.datasets.SinCurve(train=True)
    dataloader = SeqDataLoader(train_set, batch_size)
    seq_len = len(train_set)

    model = BetterRNN(hidden_size, 1)
    optim = dezero.optimizers.Adam().setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0

        for x, t in dataloader:
            y = model(x)
            loss += F.mean_squared_error(y, t)
            count += 1

            if count % bptt_length == 0 or count == seq_len:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optim.update()
        avg_loss = float(loss.data) / count
        print(f"| epoch {epoch} | loss {avg_loss}")

    # test

    xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
    model.reset_state()
    pred_list = []

    with dezero.no_grad():
        for x in xs:
            x = np.array(x).reshape(1, 1)
            y = model(x)
            pred_list.append(float(y.data.flatten()))

    plt.plot(np.arange(len(xs)), xs, label="y=cos(x)")
    plt.plot(np.arange(len(xs)), pred_list, label="predict")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    step60()
