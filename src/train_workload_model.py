import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 1. Load workload data
# --------------------------------------------------

data = pd.read_csv(
    DATA_DIR / "employee_weekly_workload.csv"
)

print("=" * 60)
print("WORKLOAD PREDICTION MODEL")
print("=" * 60)

print("\nDataset shape:", data.shape)


# --------------------------------------------------
# 2. Select features and target
# --------------------------------------------------

features = [
    "total_tasks",
    "assigned_hours",
    "actual_hours",
    "experience_years",
    "department"
]

target = "workload_level"

X = data[features]
y = data[target]


# --------------------------------------------------
# 3. Define feature types
# --------------------------------------------------

numeric_features = [
    "total_tasks",
    "assigned_hours",
    "actual_hours",
    "experience_years"
]

categorical_features = [
    "department"
]


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 5. Create machine learning model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 6. Create complete pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 7. Split data into training and testing
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
# 8. Train the model
# --------------------------------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# --------------------------------------------------
# 9. Make predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 10. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 11. Save model
# --------------------------------------------------

model_file = MODEL_DIR / "workload_prediction_model.pkl"

joblib.dump(
    pipeline,
    model_file
)

print("\nModel saved successfully!")
print(model_file)