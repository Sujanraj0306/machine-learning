import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 10
np.random.seed(42)

start_time = time.time()

models = []
for i in range(n_estimators):
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_bootstrap = X_train[indices]
    y_bootstrap = y_train[indices]
    tree = DecisionTreeClassifier()
    tree.fit(X_bootstrap, y_bootstrap)
    models.append(tree)

predictions = []
for model in models:
    preds = model.predict(X_test)
    predictions.append(preds)

predictions = np.array(predictions)

final_predictions = []
for col in predictions.T:
    counts = np.bincount(col)
    final_predictions.append(np.argmax(counts))

accuracy = accuracy_score(y_test, final_predictions)
end_time = time.time()

print(f"Bagging Accuracy: {accuracy}")
print(f"Execution Time: {end_time - start_time:.4f} seconds")

unseen_sample = np.random.rand(X.shape[1])
print("\nUnseen Sample (random):", unseen_sample)

unseen_preds = []
for model in models:
    pred = model.predict([unseen_sample])[0]
    unseen_preds.append(pred)

counts_unseen = np.bincount(unseen_preds)
final_unseen_prediction = np.argmax(counts_unseen)

print("Predictions from each model:", unseen_preds)
print("Final aggregated prediction for unseen sample:", final_unseen_prediction)
