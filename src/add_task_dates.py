import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load existing task data
tasks = pd.read_csv(DATA_DIR / "tasks.csv")

# Set random seed so results are reproducible
np.random.seed(42)

# Generate random dates for tasks
start_date = pd.Timestamp("2026-09-01")

tasks["task_date"] = start_date + pd.to_timedelta(
    np.random.randint(0, 56, size=len(tasks)),
    unit="D"
)

# Convert task date into week starting date
tasks["task_week"] = (
    tasks["task_date"]
    - pd.to_timedelta(tasks["task_date"].dt.dayofweek, unit="D")
)

# Save updated dataset
output_file = DATA_DIR / "tasks.csv"

tasks.to_csv(output_file, index=False)

print("=" * 60)
print("TASK DATE GENERATION")
print("=" * 60)

print("\nTask dates added successfully!")

print("\nTotal tasks:", len(tasks))

print("\nNew columns:")
print(tasks[["task_id", "task_date", "task_week"]].head(10))

print("\nDate range:")
print(tasks["task_date"].min(), "to", tasks["task_date"].max())

print("\nUpdated file:")
print(output_file)