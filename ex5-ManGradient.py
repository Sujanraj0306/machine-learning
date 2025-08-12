import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 5
learning_rate = 0.1

y_pred = np.full_like(y_train, np.mean(y_train), dtype=float)

models = []

for i in range(n_estimators):
    residuals = y_train - y_pred
    stump = DecisionTreeRegressor(max_depth=1)
    stump.fit(X_train, residuals)
    update = stump.predict(X_train)
    y_pred += learning_rate * update
    models.append(stump)
    mse = mean_squared_error(y_train, y_pred)
    print(f"Iteration {i+1}: MSE = {mse:.4f}")

y_test_pred = np.full_like(y_test, np.mean(y_train), dtype=float)
for model in models:
    y_test_pred += learning_rate * model.predict(X_test)

test_mse = mean_squared_error(y_test, y_test_pred)
print("\nFinal Test MSE:", test_mse)

y_test_class = np.where(y_test_pred >= 0.5, 1, 0)
accuracy = np.mean(y_test_class == y_test)
print("Classification Accuracy (from regression output):", accuracy)
