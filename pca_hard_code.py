import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# 1. Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='Target')

print("Original shape:", X.shape)

# 2. Standardize the dataset manually
X_mean = X.mean()
X_std = X.std()
X_standardized = (X - X_mean) / X_std

# 3. Compute the covariance matrix
cov_matrix = np.cov(X_standardized.T)

# 4. Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 5. Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# 6. Project data onto principal components
X_pca = np.dot(X_standardized, eigenvectors_sorted)

# 7. Create DataFrame for principal components
pc_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pc_columns)
df_pca['Target'] = y

# 8. Calculate explained variance ratio
total_variance = sum(eigenvalues_sorted)
explained_variance_ratio = eigenvalues_sorted / total_variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# 9. Plot the explained variance
plt.figure(figsize=(10, 5))
sns.barplot(x=pc_columns, y=explained_variance_ratio)
plt.plot(range(len(cumulative_variance)), cumulative_variance, marker='o', color='red', label='Cumulative Variance')
plt.title("Explained Variance by Principal Components (Hardcoded PCA)")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# 10. Print first 5 Principal Components
print("\nTop 5 Principal Components (first few rows):")
print(df_pca[pc_columns[:5]].head())

# 11. Print explained variance ratio
print("\nExplained variance ratio for each component:")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var:.4f}")
