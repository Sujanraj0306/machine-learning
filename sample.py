from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve
from sklearn.metrics import accuracy_score,classification_report,mean_squared_error
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import matplot.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
data=load_breast_cancer()
df=pd.DataFrame(data.data,columns=data.feature_names)
X=data.data
y=data.target
print(y)
print("Summary Statistics:\n", df.describe())
print("Missing Values:\n", df.isnull().sum())
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LogisticRegression()
# model.fit(X_train,y_train)

# test_acc=model.score(X_test,y_test)
# train_acc=model.score(X_train,y_train)
# print("train accuracy:",train_acc)
# print("test accuracy:",test_acc)


results = {}
learning_curves = {}

def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    results[name] = {
        "Train Acc": train_acc,
        "Test Acc": test_acc,
    }
    print(f"\n--- {name} ---")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("classification_report:\n",classification_report(y_test,model.predict(X_test)))
models = {
    "LogisticRegression":LogisticRegression(max_iter=5),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42),
}

for name, model in models.items():
    evaluate_model(model, name)

sns.countplot(x=y)
plt.title("")
plt.show()