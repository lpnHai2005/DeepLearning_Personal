import torch

print("=== CALCULATING BÀI 3 ===")
# J = 3a + 2(4b + 5cd)
# a = 2, b = 3, c = 4, d = 5

a = 2.0
b = 3.0
c = 4.0
d = 5.0

# Forward pass
u1 = 3.0 * a          # 3 * 2 = 6
u2 = 4.0 * b          # 4 * 3 = 12
u3 = c * d            # 4 * 5 = 20
u4 = 5.0 * u3         # 5 * 20 = 100
u5 = u2 + u4          # 12 + 100 = 112
u6 = 2.0 * u5         # 2 * 112 = 224
J = u1 + u6           # 6 + 224 = 230

print(f"Forward: u1={u1}, u2={u2}, u3={u3}, u4={u4}, u5={u5}, u6={u6}, J={J}")

# Backward pass
# dJ/dJ = 1
dJ_dJ = 1.0

# J = u1 + u6
dJ_du1 = dJ_dJ * 1.0  # 1
dJ_du6 = dJ_dJ * 1.0  # 1

# u6 = 2 * u5
dJ_du5 = dJ_du6 * 2.0 # 2

# u5 = u2 + u4
dJ_du2 = dJ_du5 * 1.0 # 2
dJ_du4 = dJ_du5 * 1.0 # 2

# u4 = 5 * u3
dJ_du3 = dJ_du4 * 5.0 # 10

# u1 = 3 * a
dJ_da = dJ_du1 * 3.0  # 3

# u2 = 4 * b
dJ_db = dJ_du2 * 4.0  # 8

# u3 = c * d
dJ_dc = dJ_du3 * d    # 10 * 5 = 50
dJ_dd = dJ_du3 * c    # 10 * 4 = 40

print(f"Backward: dJ/da={dJ_da}, dJ/db={dJ_db}, dJ/dc={dJ_dc}, dJ/dd={dJ_dd}")

# Direct verification:
# J = 3a + 8b + 10cd
# dJ/da = 3
# dJ/db = 8
# dJ/dc = 10d = 10*5 = 50
# dJ/dd = 10c = 10*4 = 40
print(f"Direct formula: dJ/da=3, dJ/db=8, dJ/dc=10*d={10*d}, dJ/dd=10*c={10*c}")

# Gradient descent update with alpha = 0.01
alpha_3 = 0.01
a_new = a - alpha_3 * dJ_da
b_new = b - alpha_3 * dJ_db
c_new = c - alpha_3 * dJ_dc
d_new = d - alpha_3 * dJ_dd

J_new = 3.0 * a_new + 2.0 * (4.0 * b_new + 5.0 * c_new * d_new)

print(f"Updated parameters: a'={a_new:.4f}, b'={b_new:.4f}, c'={c_new:.4f}, d'={d_new:.4f}")
print(f"J_old = {J:.4f}, J_new = {J_new:.4f}, delta_J = {J_new - J:.4f}")

# PyTorch verification
at = torch.tensor(2.0, requires_grad=True)
bt = torch.tensor(3.0, requires_grad=True)
ct = torch.tensor(4.0, requires_grad=True)
dt = torch.tensor(5.0, requires_grad=True)

u1_t = 3.0 * at
u2_t = 4.0 * bt
u3_t = ct * dt
u4_t = 5.0 * u3_t
u5_t = u2_t + u4_t
u6_t = 2.0 * u5_t
J_t = u1_t + u6_t

J_t.backward()

print(f"PyTorch Autograd: J={J_t.item()}")
print(f"grad a={at.grad.item()}, b={bt.grad.item()}, c={ct.grad.item()}, d={dt.grad.item()}")
