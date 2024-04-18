import tensorflow as tf
import numpy as np
import datetime
from util import *

K = 10
batch_size = 100000
n_gen = 100

x_min, x_max = -10, 10
name = 'relu'

# the importance allows the user to add weights to the loss function.
# this way, certain regions can be made more important.
imp = 1

# Generate data
x = np.linspace(x_min, x_max, batch_size).astype(np.float32)
y = relu(x)

# FS coding function using tf.function for performance optimization
@tf.function
def fs_coding(x, h, d, T, K):
    v = tf.identity(x) # membrane potential
    out = tf.zeros_like(x) # output
    v_reg = tf.zeros(()) # normalization factor to ensure that 'v' is limited to [-1, 1]
    z_reg = tf.zeros(()) # normalization factor when a neuron is activated
    for t in tf.range(K):
        v_scaled = (v - T[t]) / (tf.abs(v) + 1) # calculate normalized 'v'
        z = spike_function(v_scaled) # determine whether a neuron fires
        out += z * d[t] # calculate output
        v_reg += tf.reduce_mean(tf.square(tf.maximum(tf.abs(v_scaled) - 1, 0))) # increase if 'v_scaled' is not [-1, 1]
        z_reg += tf.reduce_mean(imp * z) # if fires, increase according to 'imp'
        v -= z * h[t] # if fires, subtract 'h[t]'
    return out, v_reg, z_reg

# Variables
h = tf.Variable(tf.random.uniform([K], -1.0, 0.0))
d = tf.Variable(tf.random.uniform([K], -0.5, 1.0))
T = tf.Variable(tf.random.uniform([K], -1.0, 1.0))

# Define loss function and optimizer
optimizer = tf.optimizers.Adam()

# Training step function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_approx, v_reg, z_reg = fs_coding(x, h, d, T, K)
        loss = tf.reduce_mean(imp * tf.square(y - y_approx)) + 0.1 * v_reg + 0.0 * z_reg # calculate MSE and add 'v_reg'
    gradients = tape.gradient(loss, [h, d, T])
    optimizer.apply_gradients(zip(gradients, [h, d, T])) # update 'h', 'd', 'T' using gradients
    return loss, v_reg, z_reg

# Training loop
best_loss = float('inf')
for gen in range(n_gen):
    for i in range(5000):
        loss, v_reg, z_reg = train_step(x, y)
        if i % 1000 == 0:
            print(f"Gen: {gen}, Step: {i}, Time: {datetime.datetime.now()}, Loss: {loss.numpy()}, V_reg: {v_reg.numpy()}, Z_reg: {z_reg.numpy()}")
        if loss < best_loss:
            best_loss = loss

# Output the best loss and parameters
print("Best loss:", best_loss)
print("Parameters h:", h.numpy())
print("Parameters d:", d.numpy())
print("Parameters T:", T.numpy())
