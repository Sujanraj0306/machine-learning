from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,train_test_split

iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
print("Standard deviation of CV:", cv_scores.std())

model.fit(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

train_accuracy = model.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)