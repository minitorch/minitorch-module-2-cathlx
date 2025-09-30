"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers: int):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        # backward like this doesn't work :(

        # W = self.weights.value
        # out = minitorch.zeros((x.shape[0], W.shape[1]))

        # for w_col in range(W.shape[1]):
        #     for x_row in range(x.shape[0]):
        #         for i in range(W.shape[0]):
        #             out[x_row, w_col] = out[x_row, w_col] + x[x_row, i] * W[i, w_col]
        # return out.f.add_zip(out, self.bias.value) # (50, 2), (2,)

        # x_unsqueeze_1 = x.view(x.shape[0], 1, x.shape[1]) # x: (batch_size, in_size) -> (batch_size, 1, in_size)
        # weights = self.weights.value.permute(1, 0) # weights: (in_size, out_size) -> (out_size, in_size)

        # dot_products = (x_unsqueeze_1 * weights).sum(2) # (batch_size, out_size, 1)
        # both shapes broadcast to (batch_size, out_size, in_size) under the hood
        # dims represent: batch, neurons, features
        # for each batch: x has it's in_size features (all of them) repeated out_size times (for each neuron)
        #                 weights has all of it's out_size neurons once for every of in_size features
        # x * w: -> result: (batch_size, out_size, in_size)
        # for each batch and neuron dim 2 contains {x_i * w_i} - sum along dim 2 to get dot product

        #  return dot_products.view(x.shape[0], self.out_size) + self.bias.value

        # idk why, but backward only works for one-liner
        return (x.view(x.shape[0], 1, x.shape[1]) * self.weights.value.permute(1, 0)).sum(2).view(x.shape[0], self.out_size) + self.bias.value


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
