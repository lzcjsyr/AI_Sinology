import sqlite3
import os

db_path = '/Users/dielangli/Desktop/Coding/AI汉学/data/CBDB.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = [row[0] for row in cursor.fetchall()]

print("# CBDB Database Schema Reference\n")

for table in tables:
    print(f"## Table: {table}\n")
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    
    print("| Field Name | Type | Description |")
    print("| :--- | :--- | :--- |")
    for col in columns:
        # col structure: (cid, name, type, notnull, dflt_value, pk)
        name = col[1]
        dtype = col[2]
        desc = ""
        
        # Simple heuristics for description
        if name == 'c_personid':
            desc = "Person ID (Foreign Key to BIOG_MAIN)"
        elif name.endswith('_code'):
            desc = "Code (Lookup in corresponding CODES table)"
        elif name == 'c_dy':
            desc = "Dynasty Code"
        
        print(f"| `{name}` | {dtype} | {desc} |")
    print("\n---\n")

conn.close()
