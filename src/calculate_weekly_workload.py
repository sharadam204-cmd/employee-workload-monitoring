import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

print("=" * 60)
print("WEEKLY EMPLOYEE WORKLOAD")
print("=" * 60)

# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

employees = pd.read_csv(
    DATA_DIR / "employees.csv"
)

tasks = pd.read_csv(
    DATA_DIR / "tasks.csv"
)

# Convert task_week to datetime
tasks["task_week"] = pd.to_datetime(
    tasks["task_week"]
)


# --------------------------------------------------
# 2. Group tasks by employee and week
# --------------------------------------------------

weekly_workload = tasks.groupby(
    ["employee_id", "task_week"]
).agg(
    total_tasks=("task_id", "count"),
    assigned_hours=("estimated_hours", "sum"),
    actual_hours=("actual_hours", "sum")
).reset_index()


# --------------------------------------------------
# 3. Add employee information
# --------------------------------------------------

weekly_workload = weekly_workload.merge(
    employees,
    on="employee_id",
    how="left"
)


# --------------------------------------------------
# 4. Calculate workload percentage
# --------------------------------------------------

available_hours = 40

weekly_workload["workload_percent"] = (
    weekly_workload["assigned_hours"]
    / available_hours
) * 100


# --------------------------------------------------
# 5. Classify workload
# --------------------------------------------------

def classify_workload(percent):

    if percent <= 60:
        return "Low"

    elif percent <= 85:
        return "Normal"

    elif percent <= 100:
        return "High"

    else:
        return "Overloaded"


weekly_workload["workload_level"] = (
    weekly_workload["workload_percent"]
    .apply(classify_workload)
)


# --------------------------------------------------
# 6. Sort records
# --------------------------------------------------

weekly_workload = weekly_workload.sort_values(
    ["employee_id", "task_week"]
)


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

output_file = (
    DATA_DIR / "employee_weekly_workload.csv"
)

weekly_workload.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\nWeekly workload calculation completed!")

print(
    "\nEmployee-week records:",
    len(weekly_workload)
)

print("\nWorkload distribution:")

print(
    weekly_workload[
        "workload_level"
    ].value_counts()
)

print("\nAverage workload:")

print(
    round(
        weekly_workload[
            "workload_percent"
        ].mean(),
        2
    ),
    "%"
)

print("\nSample results:")

print(
    weekly_workload[
        [
            "employee_id",
            "employee_name",
            "department",
            "task_week",
            "total_tasks",
            "assigned_hours",
            "actual_hours",
            "workload_percent",
            "workload_level"
        ]
    ].head(10)
)

print("\nFile saved:")

print(output_file)

print("\n" + "=" * 60)
print("STEP 19 COMPLETED")
print("=" * 60)