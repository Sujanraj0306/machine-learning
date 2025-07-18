import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load Dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Store results and learning curves
results = {}
learning_curves = {}

# Evaluation function
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cv_acc = cross_val_score(model, X, y, cv=5).mean()
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    results[name] = {
        "Train Acc": train_acc,
        "Test Acc": test_acc,
        "CV Acc": cv_acc,
        "Train MSE": train_mse,
        "Test MSE": test_mse
    }

    # Store learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy',train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
    learning_curves[name] = {
        "sizes": train_sizes,
        "train_mean": train_scores.mean(axis=1),
        "test_mean": test_scores.mean(axis=1)
    }

    # Print summary
    print(f"\n--- {name} ---")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Cross-Validation Accuracy: {cv_acc:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

# Models dictionary
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate all models
for name, model in models.items():
    evaluate_model(model, name)

# ------------------- Plotting -------------------

# 1. Learning Curves (Overlaid)
plt.figure(figsize=(10, 6))
for name in learning_curves:
    plt.plot(learning_curves[name]["sizes"], learning_curves[name]["test_mean"], label=name)
plt.title("Comparison of Learning Curves")
plt.xlabel("Training Set Size")
plt.ylabel("Cross-Validation Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 2. Accuracy Bar Plot
labels = list(results.keys())
train_accs = [results[k]["Train Acc"] for k in labels]
test_accs = [results[k]["Test Acc"] for k in labels]
cv_accs = [results[k]["CV Acc"] for k in labels]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, train_accs, width, label='Train Accuracy')
plt.bar(x, test_accs, width, label='Test Accuracy')
plt.bar(x + width, cv_accs, width, label='CV Accuracy')
plt.xticks(x, labels, rotation=15)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# 3. MSE Error Bar Plot
train_mses = [results[k]["Train MSE"] for k in labels]
test_mses = [results[k]["Test MSE"] for k in labels]

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, train_mses, width, label='Train MSE')
plt.bar(x + width / 2, test_mses, width, label='Test MSE')
plt.xticks(x, labels, rotation=15)
plt.ylabel("Mean Squared Error")
plt.title("MSE Comparison")
plt.legend()
plt.tight_layout()
plt.show()
