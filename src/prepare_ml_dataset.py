import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


print("=" * 60)
print("PREPARING ML DATASET")
print("=" * 60)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

employees = pd.read_csv(
    DATA_DIR / "employees.csv"
)

tasks = pd.read_csv(
    DATA_DIR / "tasks.csv"
)


# Convert dates
tasks["task_week"] = pd.to_datetime(
    tasks["task_week"]
)


# --------------------------------------------------
# 3. Create weekly task summary
# --------------------------------------------------

weekly_tasks = tasks.groupby(
    ["employee_id", "task_week"]
).agg(
    total_tasks=("task_id", "count"),
    assigned_hours=("estimated_hours", "sum"),
    actual_hours=("actual_hours", "sum")
).reset_index()


# --------------------------------------------------
# 4. Create complete employee-week combinations
# --------------------------------------------------

weeks = pd.date_range(
    tasks["task_week"].min(),
    tasks["task_week"].max(),
    freq="7D"
)

employee_ids = employees["employee_id"].unique()


employee_week = pd.MultiIndex.from_product(
    [employee_ids, weeks],
    names=["employee_id", "task_week"]
).to_frame(index=False)


# --------------------------------------------------
# 5. Add task information
# --------------------------------------------------

employee_week = employee_week.merge(
    weekly_tasks,
    on=["employee_id", "task_week"],
    how="left"
)


# Employees with no tasks get zero values
employee_week[
    [
        "total_tasks",
        "assigned_hours",
        "actual_hours"
    ]
] = employee_week[
    [
        "total_tasks",
        "assigned_hours",
        "actual_hours"
    ]
].fillna(0)


# --------------------------------------------------
# 6. Add employee information
# --------------------------------------------------

employee_week = employee_week.merge(
    employees[
        [
            "employee_id",
            "department",
            "experience_years"
        ]
    ],
    on="employee_id",
    how="left"
)


# --------------------------------------------------
# 7. Calculate current workload
# --------------------------------------------------

employee_week["workload_percent"] = (
    employee_week["assigned_hours"] / 40
) * 100


def classify_workload(percent):

    if percent <= 60:
        return "Low"

    elif percent <= 85:
        return "Normal"

    elif percent <= 100:
        return "High"

    else:
        return "Overloaded"


employee_week["workload_level"] = (
    employee_week["workload_percent"]
    .apply(classify_workload)
)


# --------------------------------------------------
# 8. Create previous-week workload
# --------------------------------------------------

employee_week = employee_week.sort_values(
    ["employee_id", "task_week"]
)


employee_week["previous_workload_percent"] = (
    employee_week
    .groupby("employee_id")["workload_percent"]
    .shift(1)
)


# Employees without previous-week data
# receive 0
employee_week["previous_workload_percent"] = (
    employee_week["previous_workload_percent"]
    .fillna(0)
)


# --------------------------------------------------
# 9. Create next-week target
# --------------------------------------------------

employee_week["next_week_workload"] = (
    employee_week
    .groupby("employee_id")["workload_level"]
    .shift(-1)
)


# --------------------------------------------------
# 10. Remove final week
# --------------------------------------------------

ml_data = employee_week.dropna(
    subset=["next_week_workload"]
).copy()


# --------------------------------------------------
# 11. Select useful ML columns
# --------------------------------------------------

ml_data = ml_data[
    [
        "employee_id",
        "task_week",
        "department",
        "experience_years",
        "total_tasks",
        "actual_hours",
        "previous_workload_percent",
        "next_week_workload"
    ]
]


# --------------------------------------------------
# 12. Save dataset
# --------------------------------------------------

output_file = (
    DATA_DIR / "ml_workload_dataset.csv"
)

ml_data.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 13. Display results
# --------------------------------------------------

print("\nEmployees:", len(employee_ids))

print("Weeks:", len(weeks))

print(
    "Complete employee-week records:",
    len(employee_week)
)

print(
    "ML records:",
    len(ml_data)
)

print("\nTarget distribution:")

print(
    ml_data["next_week_workload"]
    .value_counts()
)

print("\nSample data:")

print(
    ml_data.head(10)
)

print("\nML dataset saved successfully!")

print(output_file)