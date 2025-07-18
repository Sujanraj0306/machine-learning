import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# 1. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. Standardize features manually
X_mean = X.mean()
X_std = X.std()
X_stdzd = (X - X_mean) / X_std

# 3. Separate by class
class_labels = np.unique(y)
mean_vectors = []
for label in class_labels:
    mean_vectors.append(np.mean(X_stdzd[y == label], axis=0))

# 4. Compute within-class scatter matrix (Sw)
n_features = X.shape[1]
Sw = np.zeros((n_features, n_features))
for label, mv in zip(class_labels, mean_vectors):
    class_scatter = np.zeros((n_features, n_features))
    for row in X_stdzd[y == label].values:
        row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    Sw += class_scatter

# 5. Compute between-class scatter matrix (Sb)
overall_mean = np.mean(X_stdzd, axis=0).reshape(n_features, 1)
Sb = np.zeros((n_features, n_features))
for i, mean_vec in enumerate(mean_vectors):
    n = X_stdzd[y == i].shape[0]
    mean_vec = mean_vec.reshape(n_features, 1)
    Sb += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

# 6. Solve eigenvalue problem for inv(Sw).Sb
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

# 7. Sort eigenvectors by decreasing eigenvalues
sorted_indices = np.argsort(abs(eig_vals))[::-1]
eig_vals = eig_vals[sorted_indices]
eig_vecs = eig_vecs[:, sorted_indices]

# 8. Select top k eigenvectors (k = number of classes - 1)
k = len(class_labels) - 1
W = eig_vecs[:, :k]  # Projection matrix

# 9. Project the data onto LDA components
X_lda = X_stdzd.dot(W)

# 10. Plot the result
df_lda = pd.DataFrame(X_lda, columns=[f'LDA{i+1}' for i in range(k)])
df_lda['Target'] = y

plt.figure(figsize=(8,6))
sns.scatterplot(x='LDA1', y=None if k == 1 else 'LDA2', hue='Target', data=df_lda, palette='Set1', s=60)
plt.title("Hardcoded LDA Projection")
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Show results
print("\nFirst few rows of LDA result:")
print(df_lda.head())

print("\nEigenvalues (discriminative power of each component):")
for i in range(k):
    print(f"LDA{i+1}: {eig_vals[i].real:.4f}")
