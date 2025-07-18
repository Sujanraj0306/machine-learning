import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 1. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='Target')

print("Original shape:", X.shape)

# 2. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply LDA
lda = LDA(n_components=2)  # Max components = number of classes - 1 (2 classes → 1 component)
X_lda = lda.fit_transform(X_scaled, y)

# 4. Create DataFrame for LDA components
df_lda = pd.DataFrame(X_lda, columns=['LDA1', 'LDA2'] if X_lda.shape[1] == 2 else ['LDA1'])
df_lda['Target'] = y

# 5. Plot LDA result
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_lda, x='LDA1', y='LDA2' if 'LDA2' in df_lda.columns else None, hue='Target', palette='Set1', s=60)
plt.title("LDA Projection of Breast Cancer Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Print first few rows
print("\nLDA Transformed Data (first few rows):")
print(df_lda.head())

# 7. Print explained variance ratio
print("\nExplained Variance Ratio:")
print(lda.explained_variance_ratio_)
