import numpy as np

# Small matrix example
A = np.array([[3, 1],
              [1, 3],
              [1, 1]], dtype=float)

# Step 1: Compute A^T A and A A^T
ATA = np.dot(A.T, A)
AAT = np.dot(A, A.T)

# Step 2: Eigen-decomposition of A^T A → V and Σ
eig_vals_V, V = np.linalg.eig(ATA)

# Sort eigenvalues (descending)
sorted_idx = np.argsort(-eig_vals_V)
eig_vals_V = eig_vals_V[sorted_idx]
V = V[:, sorted_idx]

# Step 3: Singular values
singular_values = np.sqrt(eig_vals_V)

# Step 4: Compute U from A A^T
U = np.zeros((A.shape[0], len(singular_values)))
for i in range(len(singular_values)):
    U[:, i] = np.dot(A, V[:, i]) / singular_values[i]

# Step 5: Σ matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma, singular_values)

print("U:\n", U)
print("Sigma:\n", Sigma)
print("V^T:\n", V.T)

# Check reconstruction
A_reconstructed = np.dot(U, np.dot(Sigma, V.T))
print("Reconstructed A:\n", A_reconstructed)
