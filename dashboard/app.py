from pathlib import Path

import firebase_admin
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from firebase_admin import credentials, firestore


# --------------------------------------------------
# 1. Project paths and page configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_workload_prediction_model.pkl"
KEY_PATH = BASE_DIR / "serviceAccountKey.json"

st.set_page_config(
    page_title="Employee Workload Monitoring System",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Employee Workload Monitoring System")
st.caption(
    "Monitor employee workload, explore department trends, "
    "and predict next-week workload using machine learning."
)


# --------------------------------------------------
# 2. Load data from Firestore and the ML model
# --------------------------------------------------

@st.cache_resource
def get_db():
    """Connect to Firestore (local key file, or Streamlit secrets when deployed)."""
    if not firebase_admin._apps:
        if KEY_PATH.exists():
            cred = credentials.Certificate(str(KEY_PATH))  # running on your PC
        else:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))  # deployed
        firebase_admin.initialize_app(cred)
    return firestore.client()


@st.cache_data(ttl=600)
def load_collection(name):
    """Read one Firestore collection into a DataFrame (cached for 10 minutes)."""
    docs = get_db().collection(name).stream()
    return pd.DataFrame([doc.to_dict() for doc in docs])


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


try:
    workload_df = load_collection("employee_weekly_workload")
except Exception as error:
    st.error(f"Could not load workload data from Firestore: {error}")
    st.stop()

if workload_df.empty:
    st.error("The 'employee_weekly_workload' collection in Firestore is empty.")
    st.info("Run src/upload_to_firestore.py to upload the data.")
    st.stop()


if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Run src/train_final_model.py to train and save the model.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the prediction model: {error}")
    st.stop()


# --------------------------------------------------
# 3. Validate and prepare workload data
# --------------------------------------------------

required_columns = {
    "employee_id",
    "department",
    "workload_percent",
    "workload_level",
}

missing_columns = required_columns - set(workload_df.columns)

if missing_columns:
    st.error(
        "The workload data is missing these columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

workload_df["workload_percent"] = pd.to_numeric(
    workload_df["workload_percent"], errors="coerce"
)

workload_df = workload_df.dropna(
    subset=["employee_id", "department", "workload_percent", "workload_level"]
).copy()

if workload_df.empty:
    st.warning("No valid workload records are available.")
    st.stop()

if "task_week" in workload_df.columns:
    workload_df["task_week"] = pd.to_datetime(
        workload_df["task_week"], errors="coerce"
    )


# --------------------------------------------------
# 4. Sidebar filters
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

departments = sorted(
    workload_df["department"].astype(str).unique().tolist()
)

selected_departments = st.sidebar.multiselect(
    "Select departments",
    options=departments,
    default=departments,
)

if selected_departments:
    filtered_df = workload_df[
        workload_df["department"].astype(str).isin(selected_departments)
    ].copy()
else:
    filtered_df = workload_df.iloc[0:0].copy()
    st.warning("Select at least one department in the sidebar.")


# --------------------------------------------------
# 5. Summary metrics
# --------------------------------------------------

st.subheader("Workload Overview")

if not filtered_df.empty:
    average_workload = filtered_df["workload_percent"].mean()
    overloaded_count = (
        filtered_df["workload_level"].astype(str).str.lower() == "overloaded"
    ).sum()
    employee_count = filtered_df["employee_id"].nunique()
    record_count = len(filtered_df)

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Employees", f"{employee_count:,}")
    metric2.metric("Employee-Week Records", f"{record_count:,}")
    metric3.metric("Average Workload", f"{average_workload:.1f}%")
    metric4.metric("Overloaded Records", f"{overloaded_count:,}")
else:
    st.info("No records match the selected filters.")


# --------------------------------------------------
# 6. Workload distribution chart
# --------------------------------------------------

st.subheader("Workload Category Distribution")

if not filtered_df.empty:
    category_order = ["Low", "Normal", "High", "Overloaded"]

    distribution = (
        filtered_df["workload_level"]
        .value_counts()
        .reindex(category_order, fill_value=0)
        .rename_axis("Workload Category")
        .reset_index(name="Number of Records")
    )

    fig_distribution = px.bar(
        distribution,
        x="Workload Category",
        y="Number of Records",
        color="Workload Category",
        category_orders={"Workload Category": category_order},
        title="Employee-Week Records by Workload Category",
        text="Number of Records",
    )

    fig_distribution.update_layout(
        showlegend=False,
        xaxis_title="Workload Category",
        yaxis_title="Number of Records",
    )
    fig_distribution.update_traces(textposition="outside")

    st.plotly_chart(fig_distribution, width="stretch")


# --------------------------------------------------
# 7. Average workload by department
# --------------------------------------------------

st.subheader("Average Workload by Department")

if not filtered_df.empty:
    department_summary = (
        filtered_df.groupby("department", as_index=False)
        .agg(
            average_workload=("workload_percent", "mean"),
            overloaded_records=(
                "workload_level",
                lambda values: (
                    values.astype(str).str.lower() == "overloaded"
                ).sum(),
            ),
        )
        .sort_values("average_workload", ascending=False)
    )

    fig_department = px.bar(
        department_summary,
        x="department",
        y="average_workload",
        color="average_workload",
        title="Average Workload Percentage by Department",
        text=department_summary["average_workload"].map(
            lambda value: f"{value:.1f}%"
        ),
        hover_data=["overloaded_records"],
    )

    fig_department.update_layout(
        xaxis_title="Department",
        yaxis_title="Average Workload (%)",
    )
    fig_department.update_traces(textposition="outside")

    st.plotly_chart(fig_department, width="stretch")


# --------------------------------------------------
# 8. Weekly workload trend
# --------------------------------------------------

st.subheader("Weekly Workload Trend")

if "task_week" in filtered_df.columns:
    trend_df = filtered_df.dropna(subset=["task_week"])

    if not trend_df.empty:
        weekly_summary = (
            trend_df.groupby("task_week", as_index=False)
            .agg(average_workload=("workload_percent", "mean"))
            .sort_values("task_week")
        )

        fig_trend = px.line(
            weekly_summary,
            x="task_week",
            y="average_workload",
            markers=True,
            title="Average Employee Workload Over Time",
        )

        fig_trend.update_layout(
            xaxis_title="Week",
            yaxis_title="Average Workload (%)",
        )

        st.plotly_chart(fig_trend, width="stretch")
    else:
        st.info("No valid weekly dates are available for the trend chart.")
else:
    st.info("The workload data does not contain a task_week column.")


# --------------------------------------------------
# 9. Estimated versus actual task hours
# --------------------------------------------------

st.subheader("Estimated vs Actual Task Hours")

try:
    task_df = load_collection("task_performance_analysis")

    required_task_columns = {
        "employee_id",
        "estimated_hours",
        "actual_hours",
    }

    missing_task_columns = required_task_columns - set(task_df.columns)

    if task_df.empty:
        st.warning(
            "Task performance data is empty in Firestore. "
            "Other dashboard charts can still work."
        )
    elif missing_task_columns:
        st.warning(
            "Task performance data is missing columns: "
            + ", ".join(sorted(missing_task_columns))
        )
    else:
        # Get employee-to-department mapping from workload data.
        employee_departments = (
            workload_df[["employee_id", "department"]]
            .dropna()
            .drop_duplicates(subset=["employee_id"])
        )

        # Drop any existing department column so the merge does not clash.
        task_df = task_df.drop(columns=["department"], errors="ignore")

        # Add department information to each task using employee_id.
        task_df = task_df.merge(
            employee_departments,
            on="employee_id",
            how="left",
        )

        # Convert hour columns to numeric values.
        task_df["estimated_hours"] = pd.to_numeric(
            task_df["estimated_hours"], errors="coerce"
        )
        task_df["actual_hours"] = pd.to_numeric(
            task_df["actual_hours"], errors="coerce"
        )

        task_df = task_df.dropna(
            subset=[
                "department",
                "estimated_hours",
                "actual_hours",
            ]
        )

        # Apply the dashboard's department filter.
        if selected_departments:
            task_df = task_df[
                task_df["department"].astype(str).isin(selected_departments)
            ]

        if not task_df.empty:
            # Calculate average estimated and actual hours per task.
            hours_summary = (
                task_df.groupby("department", as_index=False)
                .agg(
                    estimated_hours=("estimated_hours", "mean"),
                    actual_hours=("actual_hours", "mean"),
                )
            )

            # Convert to long format for a grouped bar chart.
            hours_long = hours_summary.melt(
                id_vars="department",
                value_vars=["estimated_hours", "actual_hours"],
                var_name="Hour Type",
                value_name="Average Hours",
            )

            hours_long["Hour Type"] = hours_long["Hour Type"].map(
                {
                    "estimated_hours": "Estimated Hours",
                    "actual_hours": "Actual Hours",
                }
            )

            fig_hours = px.bar(
                hours_long,
                x="department",
                y="Average Hours",
                color="Hour Type",
                barmode="group",
                title="Average Estimated vs Actual Hours by Department",
                text_auto=".1f",
            )

            fig_hours.update_layout(
                xaxis_title="Department",
                yaxis_title="Average Hours per Task",
            )

            st.plotly_chart(fig_hours, width="stretch")
        else:
            st.info(
                "No task performance records match the selected departments."
            )

except Exception as error:
    st.error(f"Could not display task performance chart: {error}")


# --------------------------------------------------
# 10. Employee workload records
# --------------------------------------------------

st.subheader("Employee Workload Records")

if not filtered_df.empty:
    display_columns = [
        column
        for column in [
            "employee_id",
            "department",
            "task_week",
            "total_tasks",
            "assigned_hours",
            "actual_hours",
            "workload_percent",
            "workload_level",
        ]
        if column in filtered_df.columns
    ]

    search_employee = st.text_input(
        "Search by Employee ID",
        placeholder="For example: E001",
    ).strip()

    table_df = filtered_df.copy()

    if search_employee:
        table_df = table_df[
            table_df["employee_id"]
            .astype(str)
            .str.contains(search_employee, case=False, na=False)
        ]

    st.dataframe(
        table_df[display_columns].sort_values(
            by="workload_percent", ascending=False
        ),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("Employee records will appear when you select a department.")


# --------------------------------------------------
# 11. Next-week workload prediction
# --------------------------------------------------

st.divider()
st.header("🤖 Predict Next-Week Workload")
st.write(
    "Enter an employee's details to estimate their workload category "
    "for the following week."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        prediction_department = st.selectbox(
            "Department",
            options=departments,
            key="prediction_department",
        )

        experience_years = st.number_input(
            "Years of Experience",
            min_value=0,
            max_value=50,
            value=2,
            step=1,
        )

        total_tasks = st.number_input(
            "Number of Tasks This Week",
            min_value=0,
            max_value=100,
            value=6,
            step=1,
        )

    with col2:
        actual_hours = st.number_input(
            "Actual Hours Worked",
            min_value=0.0,
            max_value=168.0,
            value=40.0,
            step=1.0,
        )

        previous_workload_percent = st.number_input(
            "Previous Workload Percentage",
            min_value=0.0,
            max_value=200.0,
            value=75.0,
            step=1.0,
        )

    submitted = st.form_submit_button(
        "Predict Workload",
        type="primary",
        width="stretch",
    )


if submitted:
    input_data = pd.DataFrame(
        [
            {
                "department": prediction_department,
                "experience_years": experience_years,
                "total_tasks": total_tasks,
                "actual_hours": actual_hours,
                "previous_workload_percent": previous_workload_percent,
            }
        ]
    )

    try:
        prediction = model.predict(input_data)[0]
        prediction_text = str(prediction).replace("_", " ").title()

        st.subheader("Prediction Result")
        st.metric("Predicted Workload Category", prediction_text)

        recommendation_map = {
            "low": (
                "Workload appears low. Check whether additional suitable "
                "tasks or support activities are needed."
            ),
            "normal": (
                "Workload appears manageable. Continue monitoring tasks "
                "and actual working hours."
            ),
            "high": (
                "Workload may be high. Review deadlines and consider "
                "redistributing some tasks."
            ),
            "overloaded": (
                "Workload may be excessive. Review priorities and consider "
                "redistributing tasks to reduce employee pressure."
            ),
        }

        recommendation = recommendation_map.get(
            str(prediction).strip().lower(),
            "Review the employee's tasks and workload before making decisions.",
        )

        st.info(f"Recommendation: {recommendation}")

        st.caption(
            "This is a machine-learning estimate based on synthetic project "
            "data. It is not a guarantee of future workload."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.caption(
            "Check that the model was trained with the expected five input "
            "features and that its saved preprocessing pipeline is intact."
        )


# --------------------------------------------------
# 12. Footer
# --------------------------------------------------

st.divider()
st.caption(
    "Cloud-Based Employee Workload Monitoring System | "
    "MSc Data Science Project | Synthetic data for demonstration"
)