
import joblib
import pandas as pd
from pathlib import Path

# Find the project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "final_workload_prediction_model.pkl"

# Load the trained model
model = joblib.load(MODEL_FILE)

print("=" * 50)
print("EMPLOYEE WORKLOAD PREDICTION")
print("=" * 50)

# Get employee information
department = input(
    "Enter department (Finance, HR, IT, Marketing, Operations, Sales): "
).strip().title()

experience = float(input("Enter experience in years: "))
total_tasks = int(input("Enter number of tasks this week: "))
actual_hours = float(input("Enter actual hours worked this week: "))
previous_workload = float(
    input("Enter previous week's workload percentage: ")
)

# Arrange inputs using the exact feature names
employee_data = pd.DataFrame([{
    "department": department,
    "experience_years": experience,
    "total_tasks": total_tasks,
    "actual_hours": actual_hours,
    "previous_workload_percent": previous_workload
}])

# Predict next week's workload category
prediction = model.predict(employee_data)[0]

print("\n" + "-" * 50)
print("PREDICTED NEXT-WEEK WORKLOAD:", prediction.upper())
print("-" * 50)

if prediction == "Overloaded":
    print("Recommendation: Review the workload and consider redistributing tasks.")
elif prediction == "High":
    print("Recommendation: Monitor workload and review upcoming deadlines.")
elif prediction == "Normal":
    print("Recommendation: Workload appears manageable.")
else:
    print("Recommendation: Check whether additional tasks can be assigned appropriately.")