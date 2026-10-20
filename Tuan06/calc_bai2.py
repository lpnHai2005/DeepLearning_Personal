import numpy as np
import torch
import matplotlib.pyplot as plt

print("=== CALCULATING BÀI 2 ===")
# Data
# 1: 132, 1
# 2: 141, 1
# 3: 88, 0
# 4: 154, 1
# 5: 151, 1
# 6: 102, 0
# 7: 119, 0
# 8: 106, 0
# 9: 102, 0
# 10: 143, 1
# 11: 76, 0
# 12: 168, 1
# 13: 94, 0
# 14: 186, 1

x_raw = np.array([132, 141, 88, 154, 151, 102, 119, 106, 102, 143, 76, 168, 94, 186], dtype=float)
y2 = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1], dtype=float)
m2 = len(x_raw)

# Normalization: x' = (x - 130) / 50
x_norm = (x_raw - 130.0) / 50.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def bce(y, yhat):
    return -(y * np.log(yhat) + (1.0 - y) * np.log(1.0 - yhat))

print("Normalized x':", x_norm)

# Init: w=0, b=0, alpha=0.2
w0 = 0.0
b0 = 0.0
alpha = 0.2

# Iteration 0
z2_0 = w0 * x_norm + b0  # all 0
yhat2_0 = sigmoid(z2_0)  # all 0.5
loss2_0 = bce(y2, yhat2_0)  # -ln(0.5) = ln(2) approx 0.693147
J2_0 = np.mean(loss2_0)

grad_w_0 = np.mean((yhat2_0 - y2) * x_norm)
grad_b_0 = np.mean(yhat2_0 - y2)

w1 = w0 - alpha * grad_w_0
b1 = b0 - alpha * grad_b_0

print(f"\nIter 0: J = {J2_0:.6f}")
print(f"grad_w_0 = {grad_w_0:.6f}, grad_b_0 = {grad_b_0:.6f}")
print(f"w1 = {w1:.6f}, b1 = {b1:.6f}")

# Iteration 1
z2_1 = w1 * x_norm + b1
yhat2_1 = sigmoid(z2_1)
loss2_1 = bce(y2, yhat2_1)
J2_1 = np.mean(loss2_1)

grad_w_1 = np.mean((yhat2_1 - y2) * x_norm)
grad_b_1 = np.mean(yhat2_1 - y2)

w2 = w1 - alpha * grad_w_1
b2 = b1 - alpha * grad_b_1

print(f"\nIter 1: J = {J2_1:.6f}")
print(f"grad_w_1 = {grad_w_1:.6f}, grad_b_1 = {grad_b_1:.6f}")
print(f"w2 = {w2:.6f}, b2 = {b2:.6f}")

# Iteration 2
z2_2 = w2 * x_norm + b2
yhat2_2 = sigmoid(z2_2)
loss2_2 = bce(y2, yhat2_2)
J2_2 = np.mean(loss2_2)
print(f"\nAfter Iter 2: J = {J2_2:.6f}")

# Predictions for 125 min and 160 min
test_x_raw = np.array([125.0, 160.0])
test_x_norm = (test_x_raw - 130.0) / 50.0
test_z = w2 * test_x_norm + b2
test_yhat = sigmoid(test_z)
print(f"\nTest 125 min: x'={test_x_norm[0]:.4f}, z={test_z[0]:.6f}, yhat={test_yhat[0]:.6f}")
print(f"Test 160 min: x'={test_x_norm[1]:.4f}, z={test_z[1]:.6f}, yhat={test_yhat[1]:.6f}")

# Evaluation on 14 samples after 2 updates
pred_labels = (yhat2_2 >= 0.5).astype(int)
TP = np.sum((pred_labels == 1) & (y2 == 1))
TN = np.sum((pred_labels == 0) & (y2 == 0))
FP = np.sum((pred_labels == 1) & (y2 == 0))
FN = np.sum((pred_labels == 0) & (y2 == 1))

accuracy = (TP + TN) / m2
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nConfusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

# Decision boundary at yhat = 0.5 => z = 0 => w2 * x' + b2 = 0 => x' = -b2 / w2
x_norm_db = -b2 / w2
x_raw_db = x_norm_db * 50.0 + 130.0
print(f"Decision Boundary: x' = {x_norm_db:.6f} => x_raw = {x_raw_db:.2f} minutes")

# 1000 iterations training in Python
w_1000 = 0.0
b_1000 = 0.0
losses_1000 = []

for it in range(1000):
    z = w_1000 * x_norm + b_1000
    yhat = sigmoid(z)
    loss = np.mean(bce(y2, yhat))
    losses_1000.append(loss)
    
    gw = np.mean((yhat - y2) * x_norm)
    gb = np.mean(yhat - y2)
    w_1000 -= alpha * gw
    b_1000 -= alpha * gb

print(f"\nAfter 1000 iterations: w = {w_1000:.6f}, b = {b_1000:.6f}, loss = {losses_1000[-1]:.6f}")
