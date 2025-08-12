from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)  

rf = RandomForestClassifier(random_state=42)

# param_grid = {
#     'n_estimators': [100, 200, 300],          # Number of trees in the forest
#     'max_depth': [None, 5, 10, 20],           # Maximum depth of each tree
#     'min_samples_split': [2, 5, 10],          # Minimum samples required to split a node
#     'min_samples_leaf': [1, 2, 4],            # Minimum samples required in a leaf node
#     'max_features': ['sqrt', 'log2'],         # Number of features to consider at each split
#     'bootstrap': [True, False],               # Whether to use bootstrap sampling
#     'criterion': ['gini', 'entropy']          # Function to measure split quality
# }

#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
#grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
#best_model = grid_search.best_estimator_
best_model=rf.fit(X_train, y_train)

val_preds = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print("Validation Accuracy with Best Parameters:", val_accuracy)

# Evaluate on test set
test_preds = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print("Test Accuracy with Best Parameters:", test_accuracy)
