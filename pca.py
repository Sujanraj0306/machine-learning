import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='Target')

print("Original shape:", X.shape)

# Step 2: Standardize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA()  # All components
X_pca = pca.fit_transform(X_scaled)

# Step 4: Create DataFrame of Principal Components
pc_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pc_columns)

# Optional: Combine with target
df_pca['Target'] = y

# Step 5: Explained Variance
explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

# Step 6: Plot Explained Variance
plt.figure(figsize=(10,5))
sns.barplot(x=pc_columns, y=explained_var)
plt.plot(range(len(explained_var)), cum_var, marker='o', color='red', label='Cumulative Variance')
plt.title('Explained Variance by Principal Components')
plt.ylabel('Variance Ratio')
plt.xlabel('Principal Components')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Step 7: Print Top Principal Components
print("\nTop 5 Principal Components (first few rows):")
print(df_pca[pc_columns[:5]].head())

# Optional: Print explained variance ratio
print("\nExplained variance ratio for each component:")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var:.4f}")
