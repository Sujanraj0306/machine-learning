

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("generated_data.csv")


X = data[['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10']]
y = data['target']


x_np = X.to_numpy()
y_np = y.to_numpy()

chunk_size = 100
accuracies = []

for i in range(10):
    start = i * chunk_size
    end = start + chunk_size

    chunk_X = x_np[start:end]
    chunk_y = y_np[start:end]


    train_idx = np.random.choice(chunk_size, size=10, replace=False)
    X_train = chunk_X[train_idx]
    y_train = chunk_y[train_idx]

    global_train_idx = start + train_idx
    all_indices = np.arange(1000)
    test_idx = np.array([idx for idx in all_indices if idx not in global_train_idx])

    X_test = x_np[test_idx]
    y_test = y_np[test_idx]

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Part {i+1}: Accuracy = {acc:.4f}")


print(f"\nAverage Accuracy over 10 parts: {np.mean(accuracies):.4f}")
