import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 10
models = []
alphas = []
weights = np.ones(len(X_train)) / len(X_train)

for _ in range(n_estimators):
    stump = DecisionTreeClassifier(max_depth=1)
    stump.fit(X_train, y_train, sample_weight=weights)
    pred = stump.predict(X_train)
    err = np.sum(weights * (pred != y_train)) / np.sum(weights)
    alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
    weights *= np.exp(-alpha * y_train * pred)
    weights /= np.sum(weights)
    models.append(stump)
    alphas.append(alpha)

final_pred = np.sign(sum(alpha * model.predict(X_test) for alpha, model in zip(alphas, models)))
accuracy = accuracy_score(y_test, final_pred)

print("Boosting Accuracy:", accuracy)
