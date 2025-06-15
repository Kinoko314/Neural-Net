import numpy as np
import os

SAVE_DIR = "Neural Net/weights" # Ensure this path is correct for your system

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

def train_step(nn, inputs, action_performed, reward, learning_rate=0.1):
    # Forward pass - no change
    z1 = np.dot(inputs, nn.W1) + nn.b1
    a1 = np.tanh(z1) # Hidden layer activation
    z2 = np.dot(a1, nn.W2) + nn.b2
    current_pred = np.tanh(z2) # Output layer activation
    
    # --- Corrected Backpropagation ---

    # 1. Calculate the 'error' term for the output layer
    # This error is (target - prediction) scaled by the derivative of the output activation function (tanh)
    # The 'action_performed' here serves as the observed target.
    # We use reward to scale the update later, not the error directly here.
    output_error = (action_performed - current_pred) * (1 - current_pred**2)

    # 2. Calculate gradients for W2 and b2
    # dW2: outer product of previous layer's activation (a1) and the output error
    dW2 = np.outer(a1, output_error)
    # db2: simply the output error
    db2 = output_error

    # 3. Backpropagate the error to the hidden layer
    # This involves dot product of the output error with the transpose of W2,
    # then element-wise multiplication by the derivative of the hidden layer's activation (tanh)
    hidden_error = np.dot(output_error, nn.W2.T) * (1 - a1**2)

    # 4. Calculate gradients for W1 and b1
    # dW1: outer product of inputs and the hidden layer error
    dW1 = np.outer(inputs, hidden_error)
    # db1: simply the hidden layer error
    db1 = hidden_error

    # --- Apply weight updates, incorporating the reward ---
    # The reward should scale the actual update to the weights.
    # Positive reward reinforces the action, negative reward makes network less likely to do it.
    # A simple way to do this is to multiply the learning rate by the reward's sign and magnitude.
    
    # Ensure there's an actual update if reward is non-zero
    
    if reward != 0:
        # Scale updates by the reward. A negative reward reverses the update direction,
        # which is correct for penalizing actions. abs(reward) could also be used
        # if you only want magnitude scaling, but `reward` directly is better for direction.
        
        nn.W1 += learning_rate * dW1 * reward
        nn.b1 += learning_rate * db1 * reward
        nn.W2 += learning_rate * dW2 * reward
        nn.b2 += learning_rate * db2 * reward
        '''
        nn.W1 += learning_rate * reward
        nn.b1 += learning_rate * reward
        nn.W2 += learning_rate * reward
        nn.b2 += learning_rate * reward
        '''
    #print("\n\n", "\n\n", "\n\n", reward, "\n\n",)
