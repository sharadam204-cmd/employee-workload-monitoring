import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

np.random.seed(42)

print("=" * 60)
print("GENERATING REALISTIC WORKLOAD DATA")
print("=" * 60)

# --------------------------------------------------
# 1. Load employees and projects
# --------------------------------------------------

employees = pd.read_csv(
    DATA_DIR / "employees.csv"
)

projects = pd.read_csv(
    DATA_DIR / "projects.csv"
)

old_tasks_file = DATA_DIR / "tasks.csv"

# --------------------------------------------------
# 2. Backup existing task data
# --------------------------------------------------

backup_file = DATA_DIR / "tasks_random_backup.csv"

if old_tasks_file.exists():
    old_tasks = pd.read_csv(old_tasks_file)
    old_tasks.to_csv(
        backup_file,
        index=False
    )

    print("\nOriginal tasks backed up to:")
    print(backup_file)


# --------------------------------------------------
# 3. Create employee workload tendencies
# --------------------------------------------------

employee_base_workload = {}

for employee_id in employees["employee_id"]:

    base_workload = np.clip(
        np.random.normal(75, 25),
        20,
        130
    )

    employee_base_workload[employee_id] = (
        base_workload
    )


# --------------------------------------------------
# 4. Generate weekly workload patterns
# --------------------------------------------------

start_date = pd.Timestamp("2026-08-31")

weeks = pd.date_range(
    start=start_date,
    periods=9,
    freq="7D"
)

weekly_workload = []

for employee_id in employees["employee_id"]:

    base = employee_base_workload[
        employee_id
    ]

    previous_workload = base

    for week in weeks:

        # Previous workload influences
        # current workload.
        current_workload = (
            0.65 * previous_workload
            + 0.35 * base
            + np.random.normal(0, 12)
        )

        current_workload = np.clip(
            current_workload,
            5,
            150
        )

        weekly_workload.append({
            "employee_id": employee_id,
            "task_week": week,
            "workload_percent": current_workload
        })

        previous_workload = current_workload


weekly_workload = pd.DataFrame(
    weekly_workload
)


# --------------------------------------------------
# 5. Generate task records
# --------------------------------------------------

statuses = [
    "Completed",
    "In Progress",
    "Pending"
]

priorities = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

tasks = []

task_number = 1

for _, row in weekly_workload.iterrows():

    employee_id = row["employee_id"]
    task_week = row["task_week"]
    workload_percent = row["workload_percent"]

    # Convert workload percentage
    # into approximately required hours.
    total_hours = (
        workload_percent / 100
    ) * 40

    # Number of tasks depends on workload.
    if workload_percent < 30:
        number_of_tasks = np.random.randint(1, 3)

    elif workload_percent < 70:
        number_of_tasks = np.random.randint(2, 4)

    elif workload_percent < 100:
        number_of_tasks = np.random.randint(3, 5)

    else:
        number_of_tasks = np.random.randint(4, 7)

    # Divide total workload hours
    # among the employee's tasks.
    task_weights = np.random.dirichlet(
        np.ones(number_of_tasks)
    )

    task_hours = (
        task_weights * total_hours
    )

    for hours in task_hours:

        estimated_hours = max(
            2,
            int(round(hours))
        )

        actual_hours = max(
            1,
            int(
                round(
                    np.random.normal(
                        estimated_hours,
                        max(
                            1,
                            estimated_hours * 0.15
                        )
                    )
                )
            )
        )

        tasks.append({
            "task_id": f"T{task_number:05d}",
            "employee_id": employee_id,
            "project_id": np.random.choice(
                projects["project_id"]
            ),
            "status": np.random.choice(
                statuses
            ),
            "priority": np.random.choice(
                priorities
            ),
            "estimated_hours": estimated_hours,
            "actual_hours": actual_hours,
            "task_date": task_week + pd.to_timedelta(
                np.random.randint(0, 7),
                unit="D"
            ),
            "task_week": task_week
        })

        task_number += 1


tasks_df = pd.DataFrame(tasks)


# --------------------------------------------------
# 6. Save realistic task dataset
# --------------------------------------------------

output_file = DATA_DIR / "tasks.csv"

tasks_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\nRealistic task data generated!")

print("\nEmployees:")
print(
    len(employees)
)

print("\nWeeks:")
print(
    len(weeks)
)

print("\nEmployee-week records:")
print(
    len(weekly_workload)
)

print("\nTotal tasks:")
print(
    len(tasks_df)
)

print("\nWorkload distribution:")

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
    weekly_workload[
        "workload_percent"
    ].apply(classify_workload)
)

print(
    weekly_workload[
        "workload_level"
    ].value_counts()
)

print("\nTask date range:")

print(
    tasks_df["task_date"].min(),
    "to",
    tasks_df["task_date"].max()
)

print("\nNew task file:")
print(output_file)

print("\n" + "=" * 60)
print("STEP 18 COMPLETED")
print("=" * 60)