import pandas as pd
from pathlib import Path

# Project data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load datasets
employees = pd.read_csv(DATA_DIR / "employees.csv")
projects = pd.read_csv(DATA_DIR / "projects.csv")
tasks = pd.read_csv(DATA_DIR / "tasks.csv")

# Store datasets in a dictionary
datasets = {
    "Employees": employees,
    "Projects": projects,
    "Tasks": tasks
}

# Inspect each dataset
for name, df in datasets.items():

    print("\n" + "=" * 50)
    print(f"{name} Dataset")
    print("=" * 50)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 5 records:")
    print(df.head())