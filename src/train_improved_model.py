import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_DIR.mkdir(exist_ok=True)


print("=" * 60)
print("IMPROVED WORKLOAD PREDICTION MODEL")
print("=" * 60)


# --------------------------------------------------
# 2. Load weekly workload data
# --------------------------------------------------

data = pd.read_csv(
    DATA_DIR / "employee_weekly_workload.csv"
)

data["task_week"] = pd.to_datetime(
    data["task_week"]
)

print("\nOriginal records:", len(data))


# --------------------------------------------------
# 3. Sort data
# --------------------------------------------------

data = data.sort_values(
    ["employee_id", "task_week"]
).reset_index(drop=True)


# --------------------------------------------------
# 4. Create next week's target
# --------------------------------------------------

future_data = data[
    [
        "employee_id",
        "task_week",
        "workload_level"
    ]
].copy()

future_data["task_week"] = (
    future_data["task_week"]
    - pd.Timedelta(days=7)
)

future_data = future_data.rename(
    columns={
        "workload_level": "next_week_workload"
    }
)


# --------------------------------------------------
# 5. Connect current week with next week
# --------------------------------------------------

data = data.merge(
    future_data,
    on=["employee_id", "task_week"],
    how="left"
)


# Remove records where next week's workload
# is not available
data = data.dropna(
    subset=["next_week_workload"]
).copy()


print(
    "Records with next-week target:",
    len(data)
)


# --------------------------------------------------
# 6. Define input features
# --------------------------------------------------

features = [
    "total_tasks",
    "actual_hours",
    "experience_years",
    "department"
]

target = "next_week_workload"

X = data[features]
y = data[target]


# --------------------------------------------------
# 7. Feature types
# --------------------------------------------------

numeric_features = [
    "total_tasks",
    "actual_hours",
    "experience_years"
]

categorical_features = [
    "department"
]


# --------------------------------------------------
# 8. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 9. Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 10. Complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# --------------------------------------------------
# 11. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 12. Train model
# --------------------------------------------------

print("\nTraining improved model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")


# --------------------------------------------------
# 13. Predictions
# --------------------------------------------------

y_pred = pipeline.predict(
    X_test
)


# --------------------------------------------------
# 14. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(
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


# --------------------------------------------------
# 15. Save improved model
# --------------------------------------------------

model_file = (
    MODEL_DIR /
    "improved_workload_prediction_model.pkl"
)

joblib.dump(
    pipeline,
    model_file
)


print("\nImproved model saved successfully!")

print(model_file)