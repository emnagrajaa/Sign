import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

print("=" * 70)
print("PHASE 4: ROBUST MODEL TRAINING (ANTI-OVERFITTING)")
print("=" * 70)


def split_by_class_order(X, y, val_size=0.2, test_size=0.2):
    """Split each class by sample order to reduce near-duplicate leakage.

    For each class: earliest samples -> train, middle -> validation, latest -> test.
    """
    train_idx, val_idx, test_idx = [], [], []
    classes = np.unique(y)

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        n = len(cls_idx)

        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size)))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError("Not enough samples per class for ordered split.")

        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )


print("\n[STEP 1] Loading and exploring data...")
try:
    df = pd.read_csv("data/gestures.csv")
    print("Data loaded successfully")
except FileNotFoundError:
    print("Error: data/gestures.csv not found")
    print("Please ensure capture_data.py has been run to generate the CSV")
    raise SystemExit(1)

print(f"\nDataset Shape: {df.shape}")
print(f"  -> Total samples: {df.shape[0]}")
print(f"  -> Total columns: {df.shape[1]}")

print("\nSamples per gesture:")
print(df["label"].value_counts().sort_index())

feature_cols = [c for c in df.columns if c != "label"]
duplicates = df.duplicated(subset=feature_cols).sum()
if duplicates > 0:
    print(f"\nFound exact duplicate feature rows: {duplicates}")
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    print(f"Removed duplicates. New shape: {df.shape}")

missing_values = df.isnull().sum().sum()
print(f"\nMissing values: {missing_values}")
if missing_values > 0:
    print("Warning: Found missing values. Consider removing or imputing them.")

print("\nFirst sample:")
print(df.head(1))


print("\n[STEP 2] Creating class distribution visualization...")
plt.figure(figsize=(12, 5))
df["label"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Samples per Gesture", fontsize=14, fontweight="bold")
plt.xlabel("Gesture", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

os.makedirs("data", exist_ok=True)
plt.savefig("data/class_distribution.png", dpi=120, bbox_inches="tight")
print("Saved: data/class_distribution.png")
plt.close()


print("\n[STEP 3] Preparing features and labels...")
X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y_encoded.shape}")
print(f"Classes: {list(le.classes_)}")


print("\n[STEP 4] Splitting data into train/validation/test...")
print("Using per-class ordered split to reduce near-duplicate leakage from random mixing.")

X_train, X_val, X_test, y_train, y_val, y_test = split_by_class_order(
    X,
    y_encoded,
    val_size=0.2,
    test_size=0.2,
)

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])

print(f"Train samples:      {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")


print("\n[STEP 5] Building regularized model candidates...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42,
)

# Soft voting combines RF + GB probabilities for a more robust final model.
voting_model = VotingClassifier(
    estimators=[("rf", rf_model), ("gb", gb_model)],
    voting="soft",
    n_jobs=-1,
)

models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "RF+GB Voting": voting_model,
}

print("\n[STEP 6] Cross-validating model candidates...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for model_name, model in models.items():
    start = time.time()
    cv_scores = cross_val_score(
        model,
        X_trainval,
        y_trainval,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    elapsed = time.time() - start
    cv_results[model_name] = {
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_scores": cv_scores,
        "cv_time": elapsed,
    }
    print(
        f"{model_name:<20} | CV F1: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f} | Time: {elapsed:.1f}s"
    )

best_model_name = max(cv_results, key=lambda k: cv_results[k]["cv_mean"])
best_model = models[best_model_name]
print(f"\nSelected by CV: {best_model_name}")


print("\n[STEP 7] Training selected model on train+validation set...")
start = time.time()
best_model.fit(X_trainval, y_trainval)
train_time = time.time() - start
print(f"Training done in {train_time:.2f}s")

y_train_pred = best_model.predict(X_trainval)
y_test_pred = best_model.predict(X_test)

train_acc = accuracy_score(y_trainval, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_trainval, y_train_pred, average="weighted")
test_f1 = f1_score(y_test, y_test_pred, average="weighted")

acc_gap = train_acc - test_acc
f1_gap = train_f1 - test_f1

print("\nGeneralization check:")
print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy:  {test_acc * 100:.2f}%")
print(f"Accuracy Gap:   {acc_gap * 100:.2f}%")
print(f"Train F1:       {train_f1:.4f}")
print(f"Test F1:        {test_f1:.4f}")
print(f"F1 Gap:         {f1_gap:.4f}")

if acc_gap > 0.03:
    print("Warning: noticeable overfitting detected (gap > 3%).")
else:
    print("Good: low train-test gap, overfitting appears controlled.")

print("\nClassification report on holdout test:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_, digits=4))


print("\n[STEP 8] Creating confusion matrix for selected model...")
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cbar_kws={"label": "Count"},
)
plt.title(f"{best_model_name} - Confusion Matrix", fontsize=14, fontweight="bold")
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.tight_layout()
plt.savefig("data/confusion_matrix_best.png", dpi=120, bbox_inches="tight")
print("Saved: data/confusion_matrix_best.png")
plt.close()


print("\n[STEP 9] Saving model artifacts...")
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/gesture_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

metadata = {
    "best_model": best_model_name,
    "train_accuracy": float(train_acc),
    "test_accuracy": float(test_acc),
    "train_f1": float(train_f1),
    "test_f1": float(test_f1),
    "accuracy_gap": float(acc_gap),
    "f1_gap": float(f1_gap),
    "cv_results": {
        name: {
            "cv_mean": vals["cv_mean"],
            "cv_std": vals["cv_std"],
            "cv_time": vals["cv_time"],
        }
        for name, vals in cv_results.items()
    },
    "n_classes": len(le.classes_),
    "classes": list(le.classes_),
    "n_features": X.shape[1],
}

joblib.dump(metadata, "models/metadata.pkl")

print("Saved: models/gesture_model.pkl")
print("Saved: models/label_encoder.pkl")
print("Saved: models/metadata.pkl")


print("\n[STEP 10] Verifying saved model...")
loaded_model = joblib.load("models/gesture_model.pkl")
loaded_le = joblib.load("models/label_encoder.pkl")

sample = X_test[0].reshape(1, -1)
pred = loaded_model.predict(sample)[0]
pred_label = loaded_le.inverse_transform([pred])[0]
actual_label = loaded_le.inverse_transform([y_test[0]])[0]

print(f"Predicted: {pred_label}")
print(f"Actual:    {actual_label}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Selected Model: {best_model_name}")
print(f"Holdout Test Accuracy: {test_acc * 100:.2f}%")
print(f"Holdout Test F1: {test_f1:.4f}")
print(f"Train-Test Accuracy Gap: {acc_gap * 100:.2f}%")
print("Generated files:")
print("  - data/class_distribution.png")
print("  - data/confusion_matrix_best.png")
print("  - models/gesture_model.pkl")
print("  - models/label_encoder.pkl")
print("  - models/metadata.pkl")
