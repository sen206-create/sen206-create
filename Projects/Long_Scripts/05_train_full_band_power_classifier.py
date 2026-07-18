import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path(os.environ.get("IEEG_DATA_ROOT",
                 Path.home() / "Desktop" / "EEG data #2"))
DATASET_PATH = DATA_ROOT / "full_band_power_ml_dataset.csv"

FEATURE_COLUMNS = [
    "theta",
    "alpha",
    "beta",
    "gamma",
    "high_gamma",
]


dataset = pd.read_csv(DATASET_PATH)

X = dataset[FEATURE_COLUMNS]
y = dataset["label"]

print("Dataset shape:", dataset.shape)
print("Class counts:")
print(dataset["label_name"].value_counts())


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def report_results(title, X_train, X_test, y_train, y_test):
    model = make_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("Accuracy:", accuracy)
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(y_test, y_pred,
          target_names=["not_tap", "tap"], zero_division=0))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
report_results("Random split across all examples",
               X_train, X_test, y_train, y_test)

recording_groups = dataset["subject"] + "_" + \
    dataset["session"] + "_" + dataset["file"]

if recording_groups.nunique() > 1:
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_index, test_index = next(
        splitter.split(X, y, groups=recording_groups))

    report_results(
        "Held-out recording split",
        X.iloc[train_index],
        X.iloc[test_index],
        y.iloc[train_index],
        y.iloc[test_index],
    )
else:
    print("\nHeld-out recording split skipped: only one recording found.")


if dataset["subject"].nunique() > 1:
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_index, test_index = next(
        splitter.split(X, y, groups=dataset["subject"]))

    report_results(
        "Held-out subject split",
        X.iloc[train_index],
        X.iloc[test_index],
        y.iloc[train_index],
        y.iloc[test_index],
    )
else:
    print("\nHeld-out subject split skipped: only one subject found.")
