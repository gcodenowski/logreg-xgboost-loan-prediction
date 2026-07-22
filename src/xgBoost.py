import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Get preprocessed data
X_train, X_test, y_train, y_test = preprocess()

# Train & save the model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/xgboost_model.pkl")

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
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - XGBoost")
plt.savefig("outputs/confusion_matrix_xgboost.png", dpi=150, bbox_inches="tight")
plt.close()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(6, 5))

# Coin flip line (random classifier)
ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)", linewidth=1)

ax.plot(fpr, tpr, color='blue', label=f"XGBoost (AUC = {auc:.4f})", linewidth=2)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - XGBoost")
ax.legend(loc="lower right", bbox_to_anchor=(1.15, 0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.savefig("outputs/roc_curve_xgboost.png", dpi=150, bbox_inches="tight")
plt.close()

# Feature importance (XGBoost-specific)
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importance.sort_values(ascending=True).plot(kind="barh")
plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance")
plt.savefig("outputs/feature_importance_xgboost.png", dpi=150, bbox_inches="tight")
plt.close()

# Summary
print("\n=== XGBoost Results ===")
print(f"Accuracy:   {accuracy:.2%}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-score:   {f1:.4f}")
print(f"AUC-ROC:    {auc:.4f}")
