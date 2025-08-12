import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

# ----- Load dataset -----
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----- Sigmoid function -----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ----- Parameters -----
n_estimators = 10
learning_rate = 0.1
f_pred_train = np.zeros(len(y_train))
f_pred_test = np.zeros(len(y_test))
models = []

# ----- Boosting loop -----
for i in range(n_estimators):
    # Step 1: Probability estimates
    p_train = sigmoid(f_pred_train)
    
    # Step 2: Gradient & Hessian for logistic loss
    grad = p_train - y_train
    hess = p_train * (1 - p_train)
    
    # Step 3: Target for regression tree
    target = -grad / (hess + 1e-6)
    
    # Step 4: Fit regression tree to pseudo-residuals
    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(X_train, target)
    
    # Step 5: Update predictions
    f_pred_train += learning_rate * tree.predict(X_train)
    f_pred_test += learning_rate * tree.predict(X_test)
    
    # Save model
    models.append(tree)
    
    # Step 6: Calculate loss
    loss_train = log_loss(y_train, sigmoid(f_pred_train))
    print(f"Iteration {i+1}: Train LogLoss = {loss_train:.4f}")

# ----- Final Predictions -----
final_probs = sigmoid(f_pred_test)
final_preds = (final_probs >= 0.5).astype(int)

print("\nFinal Test LogLoss:", log_loss(y_test, final_probs))
print("Final Test Accuracy:", accuracy_score(y_test, final_preds))
