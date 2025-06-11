import numpy as np

class SimpleNN:
    def __init__(self):
        self.W1 = np.random.randn(3, 6)
        self.b1 = np.zeros(6)
        self.W2 = np.random.randn(6, 2)
        self.b2 = np.zeros(2)

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        output = np.tanh(z2)
        return output  # output[0] = horizontal move (-1 to 1), output[1] = jump signal (-1 to 1)
