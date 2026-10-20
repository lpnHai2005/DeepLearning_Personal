import numpy as np
import torch
import matplotlib.pyplot as plt

print("=== CALCULATING BÀI 1 ===")
# Data
# 1: 2.4, y=0
# 2: 3.3, y=1
# 3: 3.2, y=1
# 4: 2.5, y=0
x1 = np.array([2.4, 3.3, 3.2, 2.5])
y1 = np.array([0.0, 1.0, 1.0, 0.0])
m1 = len(x1)

theta0_0 = 0.5
theta1_0 = 0.5
alpha = 0.2

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def bce_loss(y_true, y_hat):
    # per sample
    return -(y_true * np.log(y_hat) + (1.0 - y_true) * np.log(1.0 - y_hat))

# Iteration 0
z1_0 = theta0_0 + theta1_0 * x1
yhat1_0 = sigmoid(z1_0)
loss1_0 = bce_loss(y1, yhat1_0)
J1_0 = np.mean(loss1_0)

grad_theta0_0 = np.mean(yhat1_0 - y1)
grad_theta1_0 = np.mean((yhat1_0 - y1) * x1)

theta0_1 = theta0_0 - alpha * grad_theta0_0
theta1_1 = theta1_0 - alpha * grad_theta1_0

print("Iter 0:")
for i in range(m1):
    print(f"Sample {i+1}: x={x1[i]}, y={y1[i]}, z={z1_0[i]:.6f}, yhat={yhat1_0[i]:.6f}, loss={loss1_0[i]:.6f}")
print(f"J(0) = {J1_0:.6f}")
print(f"grad_theta0 = {grad_theta0_0:.6f}, grad_theta1 = {grad_theta1_0:.6f}")
print(f"theta0_1 = {theta0_1:.6f}, theta1_1 = {theta1_1:.6f}")

# Iteration 1
z1_1 = theta0_1 + theta1_1 * x1
yhat1_1 = sigmoid(z1_1)
loss1_1 = bce_loss(y1, yhat1_1)
J1_1 = np.mean(loss1_1)

grad_theta0_1 = np.mean(yhat1_1 - y1)
grad_theta1_1 = np.mean((yhat1_1 - y1) * x1)

theta0_2 = theta0_1 - alpha * grad_theta0_1
theta1_2 = theta1_1 - alpha * grad_theta1_1

print("\nIter 1:")
for i in range(m1):
    print(f"Sample {i+1}: x={x1[i]}, y={y1[i]}, z={z1_1[i]:.6f}, yhat={yhat1_1[i]:.6f}, loss={loss1_1[i]:.6f}")
print(f"J(1) = {J1_1:.6f}")
print(f"grad_theta0 = {grad_theta0_1:.6f}, grad_theta1 = {grad_theta1_1:.6f}")
print(f"theta0_2 = {theta0_2:.6f}, theta1_2 = {theta0_2:.6f}")

# Iteration 2 loss
z1_2 = theta0_2 + theta1_2 * x1
yhat1_2 = sigmoid(z1_2)
loss1_2 = bce_loss(y1, yhat1_2)
J1_2 = np.mean(loss1_2)
print(f"\nAfter iter 2: J(2) = {J1_2:.6f}")

# Prediction for x=2.8 and x=3.5
x_test1 = np.array([2.8, 3.5])
z_test1 = theta0_2 + theta1_2 * x_test1
yhat_test1 = sigmoid(z_test1)
print(f"\nTest x=2.8: z={z_test1[0]:.6f}, yhat={yhat_test1[0]:.6f}, class={1 if yhat_test1[0]>=0.5 else 0}")
print(f"Test x=3.5: z={z_test1[1]:.6f}, yhat={yhat_test1[1]:.6f}, class={1 if yhat_test1[1]>=0.5 else 0}")
