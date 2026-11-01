import pandas as pd
import plotly.express as px
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load datasets
workload = pd.read_csv(
    DATA_DIR / "employee_weekly_workload.csv"
)

task_performance = pd.read_csv(
    DATA_DIR / "task_performance_analysis.csv"
)

employees = pd.read_csv(
    DATA_DIR / "employees.csv"
)

# Convert date column
workload["task_week"] = pd.to_datetime(
    workload["task_week"]
)

print("=" * 60)
print("WORKLOAD VISUALIZATION")
print("=" * 60)


# --------------------------------------------------
# 1. Average workload by department
# --------------------------------------------------

department_workload = (
    workload.groupby("department")["workload_percent"]
    .mean()
    .reset_index()
)

fig1 = px.bar(
    department_workload,
    x="department",
    y="workload_percent",
    title="Average Workload by Department",
    labels={
        "workload_percent": "Average Workload (%)",
        "department": "Department"
    }
)

fig1.show()


# --------------------------------------------------
# 2. Workload level distribution
# --------------------------------------------------

workload_distribution = (
    workload["workload_level"]
    .value_counts()
    .reset_index()
)

workload_distribution.columns = [
    "workload_level",
    "count"
]

fig2 = px.pie(
    workload_distribution,
    names="workload_level",
    values="count",
    title="Employee Workload Level Distribution"
)

fig2.show()


# --------------------------------------------------
# 3. Estimated vs actual hours
# --------------------------------------------------

hour_comparison = task_performance[
    ["estimated_hours", "actual_hours"]
].mean().reset_index()

hour_comparison.columns = [
    "hour_type",
    "average_hours"
]

hour_comparison["hour_type"] = (
    hour_comparison["hour_type"]
    .replace({
        "estimated_hours": "Estimated Hours",
        "actual_hours": "Actual Hours"
    })
)

fig3 = px.bar(
    hour_comparison,
    x="hour_type",
    y="average_hours",
    title="Average Estimated vs Actual Task Hours",
    labels={
        "hour_type": "Hour Type",
        "average_hours": "Average Hours"
    }
)

fig3.show()


# --------------------------------------------------
# 4. Weekly workload trend
# --------------------------------------------------

weekly_average = (
    workload.groupby("task_week")["workload_percent"]
    .mean()
    .reset_index()
)

fig4 = px.line(
    weekly_average,
    x="task_week",
    y="workload_percent",
    markers=True,
    title="Weekly Average Workload Trend",
    labels={
        "task_week": "Week",
        "workload_percent": "Average Workload (%)"
    }
)

fig4.show()


print("\nVisualization completed successfully!")