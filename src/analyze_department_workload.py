import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load weekly workload data
workload = pd.read_csv(
    DATA_DIR / "employee_weekly_workload.csv"
)

print("=" * 60)
print("DEPARTMENT WORKLOAD ANALYSIS")
print("=" * 60)

# 1. Average workload percentage by department
department_average = workload.groupby("department").agg(
    average_workload=("workload_percent", "mean"),
    total_tasks=("total_tasks", "sum"),
    assigned_hours=("assigned_hours", "sum")
).reset_index()

print("\n1. Average workload by department:")
print(department_average)


# 2. Workload level distribution by department
department_distribution = pd.crosstab(
    workload["department"],
    workload["workload_level"]
)

print("\n2. Workload level distribution:")
print(department_distribution)


# 3. Number of overloaded records by department
overloaded = workload[
    workload["workload_level"] == "Overloaded"
]

overloaded_by_department = (
    overloaded.groupby("department")
    .size()
    .reset_index(name="overloaded_records")
    .sort_values(
        "overloaded_records",
        ascending=False
    )
)

print("\n3. Overloaded records by department:")
print(overloaded_by_department)


# 4. Department with highest average workload
highest_department = department_average.loc[
    department_average["average_workload"].idxmax()
]

print("\n4. Department with highest average workload:")

print(
    highest_department["department"],
    "->",
    round(highest_department["average_workload"], 2),
    "%"
)


# Save analysis results
department_average.to_csv(
    DATA_DIR / "department_workload_analysis.csv",
    index=False
)

print("\nAnalysis saved successfully!")

print(
    DATA_DIR / "department_workload_analysis.csv"
)