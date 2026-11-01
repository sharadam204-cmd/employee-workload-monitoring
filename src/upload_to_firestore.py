import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# Paths are built from this file's location, so it works from any folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

cred = credentials.Certificate(os.path.join(ROOT, "serviceAccountKey.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# CSV file  ->  Firestore collection name
FILES = {
    "employees.csv": "employees",
    "employee_workload.csv": "employee_workload",
    "employee_weekly_workload.csv": "employee_weekly_workload",
    "department_workload_analysis.csv": "department_workload_analysis",
    "task_performance_analysis.csv": "task_performance_analysis",
    "projects.csv": "projects",
    "tasks.csv": "tasks",
}

def upload_csv(file_name, collection_name):
    df = pd.read_csv(os.path.join(DATA, file_name))
    df = df.astype(object).where(df.notnull(), None)   # NaN -> None

    batch = db.batch()
    count = 0
    for i, record in enumerate(df.to_dict("records")):
        # Fixed document IDs: running the script again overwrites, never duplicates
        doc_ref = db.collection(collection_name).document(f"{i:06d}")
        batch.set(doc_ref, record)
        count += 1
        if count % 400 == 0:          # Firestore allows max 500 writes per batch
            batch.commit()
            batch = db.batch()
    batch.commit()
    print(f"Uploaded {len(df)} rows from {file_name} -> '{collection_name}'")

for csv_name, collection in FILES.items():
    upload_csv(csv_name, collection)

print("Done.")