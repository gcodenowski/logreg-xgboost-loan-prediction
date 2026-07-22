import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        ConfusionMatrixDisplay
        )
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import joblib

# Get preprocessed data
X_train, X_test, y_train, y_test = preprocess()

# Train & save the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

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
auc = roc_auc_score(y_test, y_proba)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Rejected", "Approved"])

disp.plot(cmap=plt.cm.Greens)
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("outputs/confusion_matrix_logreg.png", dpi=150, bbox_inches="tight")
plt.close()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(6, 5))

# Coin flip line (random classifier)
ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)", linewidth=1)

ax.plot(fpr, tpr, color='green', label=f"Logistic Regression (AUC = {auc:.4f})", linewidth=2)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Logistic Regression")
ax.legend(loc="lower right")  # ← this shows the coin flip label
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.savefig("outputs/roc_curve_logreg.png", dpi=150, bbox_inches="tight")  # ← THIS is what saves it
plt.close()

# Summary
print("\n=== Logistic Regression Results ===")
print(f"Accuracy:   {accuracy:.2%}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-score:   {f1:.4f}")
print(f"AUC-ROC:    {auc:.4f}")
