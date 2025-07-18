# Q1. Basic AUC-ROC Implementation
# a) Load any binary classification dataset (e.g., Breast Cancer).
# b) Train any classifier (Logistic Regression).
# c) Predict probabilities and plot the ROC curve.
# d) Calculate the AUC score.
# e) Comment on the model's performance.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict Probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Model performance report
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc)




# Q2: Comparison with Accuracy on Imbalanced Data
# a) Create an imbalanced dataset
# b) Train classifier and evaluate using Accuracy, Precision, Recall, F1, AUC-ROC
# c) Show why accuracy is misleading
# d) Plot ROC curve and confusion matrix

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Create imbalanced dataset (90% of class 0)
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],n_features=20, n_informative=2, random_state=42)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_probs))

# Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot()
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid()
plt.show()




# Q3: Compare Logistic Regression, Random Forest, SVM using AUC
from sklearn.svm import SVC

# Logistic Regression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]

# SVM with probability
svm = SVC(probability=True)
svm.fit(X_train, y_train)
svm_probs = svm.predict_proba(X_test)[:, 1]

# Plot ROC Curves
plt.figure(figsize=(8,6))
for model_name, probs in zip(["Logistic Regression", "Random Forest", "SVM"],[lr_probs, rf_probs, svm_probs]):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid()
plt.show()






# Q4: Cross-Validation for AUC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Use cross_val_score with roc_auc
cv_scores = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring='roc_auc')

print("Cross-Validation AUC Scores:", cv_scores)
print("Mean AUC:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())








# Q5: Tuning thresholds
import numpy as np
from sklearn.metrics import confusion_matrix

# Use Random Forest again
clf.fit(X_train, y_train)
y_probs = clf.predict_proba(X_test)[:, 1]

# Define thresholds
thresholds = [0.3, 0.5, 0.7]

for thresh in thresholds:
    y_thresh_pred = (y_probs >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_thresh_pred)
    print(f"\nThreshold: {thresh}")
    print("Confusion Matrix:")
    print(cm)
    print("Precision:", precision_score(y_test, y_thresh_pred))
    print("Recall:", recall_score(y_test, y_thresh_pred))
    print("F1-Score:", f1_score(y_test, y_thresh_pred))






# Q6: Convert multiclass to binary and apply AUC
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Convert to binary: 0 vs rest
y_binary = (y == 0).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:, 1]

# ROC + AUC
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

# Plot
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve (Binary: 0 vs Rest)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid()
plt.show()

print("AUC Score:", auc_score)




# Q7: Fix class imbalance and compare AUC
from imblearn.over_sampling import SMOTE

# Original AUC
clf.fit(X_train, y_train)
original_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
print("Original AUC:", original_auc)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Retrain model
clf.fit(X_resampled, y_resampled)
new_probs = clf.predict_proba(X_test)[:, 1]
new_auc = roc_auc_score(y_test, new_probs)

# ROC comparison
fpr1, tpr1, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
fpr2, tpr2, _ = roc_curve(y_test, new_probs)

plt.plot(fpr1, tpr1, label=f"Before SMOTE (AUC={original_auc:.2f})")
plt.plot(fpr2, tpr2, label=f"After SMOTE (AUC={new_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve Before vs After SMOTE")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid()
plt.show()


