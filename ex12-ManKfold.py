import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

k = 5  # Number of folds
fold_size = len(X) // k
indices = np.arange(len(X))

np.random.shuffle(indices)  # Shuffle data

accuracies = []

for i in range(k):
    # Split train/test manually
    test_idx = indices[i*fold_size:(i+1)*fold_size]
    train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {i+1} Accuracy: {acc:.4f}")

print("Average Accuracy:", np.mean(accuracies))
