"""
8-ary 3D Signal Simulation with Decision Boundaries
---------------------------------------------------
Simulates 8-ary (3D cube) constellation over AWGN channel.
Includes symbol error computation, noise visualization,
received signal plots, and decision boundary demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ----------------------
# Step 0: System Parameters
# ----------------------
A = 0.01        # Cube amplitude (half the side length)
M = 8           # Number of symbols (cube vertices)
N = 10000       # Number of transmitted symbols
N0 = 2e-4       # Noise power spectral density

noise_var = N0 / 2
sigma = np.sqrt(noise_var)

print("=== Step 0: System Parameters ===")
print(f"A = {A}, M = {M}, N = {N}")
print(f"N0 = {N0}, Noise variance = {noise_var:.6e}, σ = {sigma:.6e}\n")

# ----------------------
# Step 1: Generate Constellation
# ----------------------
# All combinations of ±A in 3D → cube centered at origin
s_m_coords_list = list(product([-A, A], repeat=3))
constellation_points = np.array(s_m_coords_list)

print("=== Step 1: Constellation Points ===")
print(constellation_points, "\n")

# ----------------------
# Step 2: Random Transmission
# ----------------------
tx_indices = np.random.randint(0, M, N)
tx_symbols = constellation_points[tx_indices]

print("=== Step 2: First 5 Transmitted Symbols ===")
print(tx_symbols[:5], "\n")

# ----------------------
# Step 3: Add AWGN Noise
# ----------------------
noise = np.random.normal(0, sigma, tx_symbols.shape)
rx_symbols = tx_symbols + noise

print("=== Step 3: First 5 Received Symbols ===")
print(rx_symbols[:5], "\n")

# ----------------------
# Step 4: ML Detection (Minimum Distance)
# ----------------------
def ml_detect(received, constellation):
    detected = []
    for r in received:
        distances = np.linalg.norm(constellation - r, axis=1)
        detected.append(np.argmin(distances))
    return np.array(detected)

rx_indices = ml_detect(rx_symbols, constellation_points)

print("=== Step 4: First 5 Detected Indices ===")
print(rx_indices[:5], "\n")

# ----------------------
# Step 5: Symbol Error Probability
# ----------------------
symbol_errors = np.sum(tx_indices != rx_indices)
Ps = symbol_errors / N

print(f"=== Step 5: Symbol Error Probability ===")
print(f"Symbol errors = {symbol_errors}/{N}")
print(f"P_symbol_error = {Ps:.6f}\n")

# ----------------------
# Step 6: Noise Distribution Visualization
# ----------------------
plt.figure(figsize=(8,5))
plt.hist(noise.flatten(), bins=50, density=True, color='green', alpha=0.7)
plt.title("Noise Distribution (All Dimensions)")
plt.xlabel("Noise Value")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# ----------------------
# Step 7: Received vs Ideal (2D projections)
# ----------------------
plt.figure(figsize=(8,6))
plt.scatter(tx_symbols[:,0], tx_symbols[:,1], c='red', s=50, label='Ideal')
plt.scatter(rx_symbols[:,0], rx_symbols[:,1], c='blue', alpha=0.3, s=10, label='Received')
plt.title("Received vs Ideal Symbols (X-Y Plane)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------
# Step 8: 3D Visualization
# ----------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(constellation_points[:,0], constellation_points[:,1], constellation_points[:,2],
           c='red', s=100, label='Constellation')
ax.scatter(rx_symbols[:2000,0], rx_symbols[:2000,1], rx_symbols[:2000,2],
           c='blue', s=5, alpha=0.3, label='Received')
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("3D Received Symbols vs Constellation")
ax.legend()
plt.show()

# ----------------------
# Step 9: Decision Boundary (2D slice visualization)
# ----------------------
# We take a 2D slice at z = 0 for visualization.
# Decision boundaries for a cube are planes at midpoints between ±A.
# That means x=0, y=0, z=0 are the boundaries dividing 8 regions.
grid_range = np.linspace(-2*A, 2*A, 200)
X, Y = np.meshgrid(grid_range, grid_range)

# For a 2D slice (z=0), find nearest constellation point
Z = np.zeros_like(X)
r_points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
dists = np.linalg.norm(r_points[:, None, :] - constellation_points[None, :, :], axis=2)
region_map = np.argmin(dists, axis=1).reshape(X.shape)

plt.figure(figsize=(8,6))
plt.contourf(X, Y, region_map, levels=8, alpha=0.4, cmap='tab10')
plt.scatter(constellation_points[:,0], constellation_points[:,1], c='red', s=80, label='Constellation')
plt.title("Decision Regions (2D slice at z=0)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

print("=== Step 9: Decision Boundary Visualization Complete ===")
print("Boundaries are planes at x=0, y=0, and z=0 dividing the cube regions.\n")

