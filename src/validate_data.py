import pandas as pd
from pathlib import Path

# Data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load datasets
employees = pd.read_csv(DATA_DIR / "employees.csv")
projects = pd.read_csv(DATA_DIR / "projects.csv")
tasks = pd.read_csv(DATA_DIR / "tasks.csv")

print("=" * 50)
print("DATA VALIDATION")
print("=" * 50)

# 1. Check duplicate IDs
print("\n1. Duplicate IDs")

print("Duplicate employee IDs:",
      employees["employee_id"].duplicated().sum())

print("Duplicate project IDs:",
      projects["project_id"].duplicated().sum())

print("Duplicate task IDs:",
      tasks["task_id"].duplicated().sum())


# 2. Check employee references
invalid_employees = tasks[
    ~tasks["employee_id"].isin(employees["employee_id"])
]

print("\n2. Invalid employee references:",
      len(invalid_employees))


# 3. Check project references
invalid_projects = tasks[
    ~tasks["project_id"].isin(projects["project_id"])
]

print("3. Invalid project references:",
      len(invalid_projects))


# 4. Check numerical values
print("\n4. Numerical value checks")

print("Invalid experience values:",
      ((employees["experience_years"] < 0) |
       (employees["experience_years"] > 50)).sum())

print("Invalid progress values:",
      ((projects["progress_percent"] < 0) |
       (projects["progress_percent"] > 100)).sum())

print("Invalid estimated hours:",
      (tasks["estimated_hours"] <= 0).sum())

print("Invalid actual hours:",
      (tasks["actual_hours"] <= 0).sum())


# Final result
print("\n" + "=" * 50)
print("VALIDATION COMPLETED")
print("=" * 50)