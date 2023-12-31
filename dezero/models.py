from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L


class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


class MLP(Model):
    def __init__(self, fc_output_size, activation=F.sigmoid):
        """
        Multi Layer Perceptron
        :param fc_output_size: Fully Connected output size
        :param activation: Activation function
        """
        super().__init__()
        self.activation= activation
        self.layers =[]

        for i, out_size in enumerate(fc_output_size):
            layer = L.Linear(out_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]: # 末尾1つのLayerを除いてループ
            x = self.activation(l(x))
        return self.layers[-1](x)
