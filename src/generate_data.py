import pandas as pd
import numpy as np
from pathlib import Path

# Make results reproducible
np.random.seed(42)

# -----------------------------
# Create data folder path
# -----------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# -----------------------------
# 1. EMPLOYEE DATA
# -----------------------------

departments = [
    "IT",
    "Finance",
    "HR",
    "Sales",
    "Marketing",
    "Operations"
]

roles = [
    "Software Developer",
    "Data Analyst",
    "Business Analyst",
    "Project Manager",
    "HR Executive",
    "QA Engineer"
]

employees = []

for i in range(1, 101):
    employees.append({
        "employee_id": f"E{i:03d}",
        "employee_name": f"Employee_{i:03d}",
        "department": np.random.choice(departments),
        "role": np.random.choice(roles),
        "experience_years": np.random.randint(1, 11)
    })

employees_df = pd.DataFrame(employees)

employees_df.to_csv(DATA_DIR / "employees.csv", index=False)

# -----------------------------
# 2. PROJECT DATA
# -----------------------------

project_names = [
    "Banking Application",
    "E-Commerce Platform",
    "Healthcare Management System",
    "Sales Dashboard",
    "Customer Management System",
    "Inventory Management System",
    "Mobile Banking App",
    "HR Management System",
    "Data Analytics Platform",
    "Online Learning Platform",
    "Insurance Management System",
    "Travel Booking System",
    "Food Delivery Platform",
    "CRM Application",
    "Financial Analytics System",
    "Employee Portal",
    "Payment Gateway",
    "Supply Chain System",
    "Marketing Analytics Platform",
    "Cloud Migration Project"
]

projects = []

for i in range(1, 21):
    projects.append({
        "project_id": f"P{i:03d}",
        "project_name": project_names[i - 1],
        "project_manager": f"E{np.random.randint(1, 101):03d}",
        "progress_percent": np.random.randint(20, 101),
        "days_remaining": np.random.randint(5, 121),
        "team_size": np.random.randint(3, 16)
    })

projects_df = pd.DataFrame(projects)

projects_df.to_csv(DATA_DIR / "projects.csv", index=False)

# -----------------------------
# 3. TASK DATA
# -----------------------------

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

for i in range(1, 501):

    estimated_hours = np.random.randint(4, 41)

    # Actual hours are related to estimated hours
    actual_hours = max(
        1,
        int(np.random.normal(estimated_hours, estimated_hours * 0.25))
    )

    tasks.append({
        "task_id": f"T{i:04d}",
        "employee_id": f"E{np.random.randint(1, 101):03d}",
        "project_id": f"P{np.random.randint(1, 21):03d}",
        "status": np.random.choice(statuses),
        "priority": np.random.choice(priorities),
        "estimated_hours": estimated_hours,
        "actual_hours": actual_hours
    })

tasks_df = pd.DataFrame(tasks)

tasks_df.to_csv(DATA_DIR / "tasks.csv", index=False)

# -----------------------------
# SUCCESS MESSAGE
# -----------------------------

print("Data generation completed successfully!")
print()
print(f"Employees created: {len(employees_df)}")
print(f"Projects created: {len(projects_df)}")
print(f"Tasks created: {len(tasks_df)}")
print()
print("Files saved in:")
print(DATA_DIR)