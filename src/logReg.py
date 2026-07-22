import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import X_test, X_train, preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        ConfusionMatrixDisplay, RocCurveDisplay
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

