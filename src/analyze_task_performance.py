import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load task data
tasks = pd.read_csv(DATA_DIR / "tasks.csv")

print("=" * 60)
print("TASK PERFORMANCE ANALYSIS")
print("=" * 60)

# Calculate difference between actual and estimated hours
tasks["time_difference"] = (
    tasks["actual_hours"] - tasks["estimated_hours"]
)

# Calculate percentage difference
tasks["time_difference_percent"] = (
    tasks["time_difference"]
    / tasks["estimated_hours"]
) * 100

# Classify task performance
def classify_performance(difference):

    if difference > 0:
        return "Over Estimated Time"

    elif difference < 0:
        return "Under Estimated Time"

    else:
        return "Exactly Estimated"


tasks["performance"] = tasks["time_difference"].apply(
    classify_performance
)

# Display overall performance distribution
print("\n1. Task performance distribution:")
print(tasks["performance"].value_counts())

# Calculate overall averages
print("\n2. Overall hour comparison:")

print(
    "Average estimated hours:",
    round(tasks["estimated_hours"].mean(), 2)
)

print(
    "Average actual hours:",
    round(tasks["actual_hours"].mean(), 2)
)

print(
    "Average time difference:",
    round(tasks["time_difference"].mean(), 2)
)

# Department-level analysis
department_performance = tasks.merge(
    pd.read_csv(DATA_DIR / "employees.csv")[
        ["employee_id", "department"]
    ],
    on="employee_id",
    how="left"
)

department_summary = department_performance.groupby(
    "department"
).agg(
    average_estimated_hours=("estimated_hours", "mean"),
    average_actual_hours=("actual_hours", "mean"),
    average_time_difference=("time_difference", "mean")
).reset_index()

print("\n3. Performance by department:")
print(department_summary)

# Save results
output_file = DATA_DIR / "task_performance_analysis.csv"

tasks.to_csv(output_file, index=False)

print("\nAnalysis saved successfully!")
print(output_file)
