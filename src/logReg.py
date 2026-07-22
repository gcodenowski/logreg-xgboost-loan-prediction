import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import X_test, X_train, preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        ConfusionMatrixDisplay
        )
import matplotlib.pyplot as plt
import joblib

# Get preprocessed data
X_train, X_test, y_train, y_test = preprocess()

# Train & save the model
model = LogisticRegression(max_iter=1000)
model.fix(X_train, y_train)

os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/logreg_model.pkl")

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_proba)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Rejected", "Approved"])

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("outputs/confusion_matrix_logreg.png", dpi=150, bbox_inches="tight")
plt.close()

# Summary
print("\n=== Logistic Regression Results ===")
print(f"Accuracy:   {accuracy:.2f}%")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-score:   {f1:.4f}")
print(f"AUC-ROC:    {auc:.4f}")
