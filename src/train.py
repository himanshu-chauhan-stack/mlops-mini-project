
# src/train.py
# training script for iris classification model

import os
import joblib
import logging
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# set up logging so we can see what is happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

def train():
    log.info("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    log.info(f"Dataset shape: {X.shape} | Classes: {iris.target_names}")

    # 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # train model
    log.info("Training Logistic Regression model...")
    model = LogisticRegression(solver="lbfgs", max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    log.info(f"Model Accuracy: {acc * 100:.2f}%")
    log.info("\n" + classification_report(y_test, preds, target_names=iris.target_names))

    # save model and class names
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/iris_model.joblib")
    np.save("models/target_names.npy", iris.target_names)
    log.info("Model saved to models/iris_model.joblib")

    return acc

if __name__ == "__main__":
    acc = train()
    print(f"\nTraining complete. Final Accuracy: {acc*100:.2f}%")
