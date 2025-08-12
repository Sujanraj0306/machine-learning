import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)  

def manual_random_forest(X_train, y_train, X_test, n_trees=10, criterion='gini'):
    trees = []
    n_samples = X_train.shape[0]
    for _ in range(n_trees):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_sample = X_train[idx]
        y_sample = y_train[idx]
        tree = DecisionTreeClassifier(criterion=criterion, max_features='sqrt', random_state=None)
        # Available parameters: criterion=['gini','entropy','log_loss'], splitter=['best','random'], max_depth, min_samples_split, min_samples_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, class_weight
        tree.fit(X_sample, y_sample)
        trees.append(tree)
    return trees

def predict_forest(trees, X):
    tree_preds = np.array([tree.predict(X) for tree in trees])
    final_preds = []
    for col in tree_preds.T:
        values, counts = np.unique(col, return_counts=True)
        final_preds.append(values[np.argmax(counts)])
    return np.array(final_preds)

start_time = time.time()
trees_gini = manual_random_forest(X_train, y_train, X_val, n_trees=10, criterion='gini')
val_preds_gini = predict_forest(trees_gini, X_val)
val_accuracy_gini = accuracy_score(y_val, val_preds_gini)
end_time = time.time()
print(f"Gini Index - Validation Accuracy: {val_accuracy_gini:.4f}, Execution Time: {end_time - start_time:.4f} seconds")

start_time = time.time()
trees_entropy = manual_random_forest(X_train, y_train, X_val, n_trees=10, criterion='entropy')
val_preds_entropy = predict_forest(trees_entropy, X_val)
val_accuracy_entropy = accuracy_score(y_val, val_preds_entropy)
end_time = time.time()
print(f"Entropy - Validation Accuracy: {val_accuracy_entropy:.4f}, Execution Time: {end_time - start_time:.4f} seconds")

test_preds_gini = predict_forest(trees_gini, X_test)
test_accuracy_gini = accuracy_score(y_test, test_preds_gini)
print(f"Gini Index - Test Accuracy: {test_accuracy_gini:.4f}")

test_preds_entropy = predict_forest(trees_entropy, X_test)
test_accuracy_entropy = accuracy_score(y_test, test_preds_entropy)
print(f"Entropy - Test Accuracy: {test_accuracy_entropy:.4f}")
