import numpy as np
import os

SAVE_DIR = "weights"

def save_weights(nn, suffix=""):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"ai_weights{suffix}.npz")
    np.savez(path, W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2)

def load_weights(nn, suffix=""):
    path = os.path.join(SAVE_DIR, f"ai_weights{suffix}.npz")
    if os.path.exists(path):
        data = np.load(path)
        nn.W1 = data['W1']
        nn.b1 = data['b1']
        nn.W2 = data['W2']
        nn.b2 = data['b2']

def train_step(nn, inputs, outputs, reward, learning_rate=0.01):
    # Forward pass
    z1 = np.dot(inputs, nn.W1) + nn.b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, nn.W2) + nn.b2
    pred = np.tanh(z2)

    # Loss is reward-based: push toward actual output if reward is positive
    error = (outputs - pred) * reward

    # Backpropagation
    dW2 = np.outer(a1, error * (1 - pred**2))
    db2 = error * (1 - pred**2)

    da1 = np.dot(nn.W2, error * (1 - pred**2))
    dW1 = np.outer(inputs, da1 * (1 - a1**2))
    db1 = da1 * (1 - a1**2)

    # Update weights
    nn.W2 += learning_rate * dW2
    nn.b2 += learning_rate * db2
    nn.W1 += learning_rate * dW1
    nn.b1 += learning_rate * db1
