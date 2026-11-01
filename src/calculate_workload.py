import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load datasets
employees = pd.read_csv(DATA_DIR / "employees.csv")
tasks = pd.read_csv(DATA_DIR / "tasks.csv")

# Calculate total estimated hours for each employee
workload = tasks.groupby("employee_id").agg(
    total_tasks=("task_id", "count"),
    assigned_hours=("estimated_hours", "sum"),
    actual_hours=("actual_hours", "sum")
).reset_index()

# Add employee information
workload = workload.merge(
    employees,
    on="employee_id",
    how="left"
)

# Assume 40 working hours available per week
available_hours = 40

# Calculate workload percentage
workload["workload_percent"] = (
    workload["assigned_hours"] / available_hours
) * 100

# Classify workload
def classify_workload(percent):

    if percent <= 60:
        return "Low"

    elif percent <= 85:
        return "Normal"

    elif percent <= 100:
        return "High"

    else:
        return "Overloaded"


workload["workload_level"] = workload["workload_percent"].apply(
    classify_workload
)

# Save workload dataset
output_file = DATA_DIR / "employee_workload.csv"

workload.to_csv(output_file, index=False)

print("=" * 60)
print("EMPLOYEE WORKLOAD CALCULATION")
print("=" * 60)

print("\nWorkload calculation completed successfully!")

print("\nEmployees analyzed:", len(workload))

print("\nWorkload distribution:")
print(workload["workload_level"].value_counts())

print("\nSample results:")
print(
    workload[
        [
            "employee_id",
            "employee_name",
            "department",
            "total_tasks",
            "assigned_hours",
            "workload_percent",
            "workload_level"
        ]
    ].head(10)
)

print("\nFile saved:")
print(output_file)