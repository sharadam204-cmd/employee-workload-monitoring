import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

print("=" * 60)
print("FINAL WORKLOAD PREDICTION MODEL")
print("=" * 60)

# --------------------------------------------------
# 1. Load ML dataset
# --------------------------------------------------

data = pd.read_csv(
    DATA_DIR / "ml_workload_dataset.csv"
)

data["task_week"] = pd.to_datetime(
    data["task_week"]
)

print("\nTotal records:", len(data))


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

features = [
    "department",
    "experience_years",
    "total_tasks",
    "actual_hours",
    "previous_workload_percent"
]

target = "next_week_workload"

X = data[features]
y = data[target]


# --------------------------------------------------
# 3. Time-based train/test split
# --------------------------------------------------

split_date = pd.Timestamp("2026-10-12")

train_data = data[
    data["task_week"] < split_date
]

test_data = data[
    data["task_week"] >= split_date
]

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 4. Encode department
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "department",
            OneHotEncoder(handle_unknown="ignore"),
            ["department"]
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 5. Create Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# --------------------------------------------------
# 6. Create complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed!")


# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

y_pred = pipeline.predict(
    X_test
)


# --------------------------------------------------
# 9. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

import joblib

MODEL_DIR.mkdir(exist_ok=True)

model_file = (
    MODEL_DIR /
    "final_workload_prediction_model.pkl"
)

joblib.dump(
    pipeline,
    model_file
)

print("\nModel saved successfully!")

print(model_file)

print("\n" + "=" * 60)
print("STEP 17 COMPLETED")
print("=" * 60)