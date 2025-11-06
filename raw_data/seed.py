import json
import psycopg2

# === Configuration ===
DB_CONFIG = {
    "dbname": "search_neu_agentic",
    "user": "postgres",
    "password": "12345678", 
    "host": "localhost",
    "port": "5432"
}

JSON_FILE = "raw_data.json"

# === Connect to PostgreSQL ===
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# === Create table (no term columns) ===
cur.execute("""
CREATE TABLE IF NOT EXISTS courses (
    id SERIAL PRIMARY KEY,
    subject TEXT,
    number INTEGER,
    title TEXT,
    description TEXT,
    min_credits INTEGER,
    max_credits INTEGER,
    instructors TEXT[]
);
""")

# === Load JSON ===
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# The JSON root contains term info + a courses list
for course in data["courses"]:
    subject = course.get("subject")
    number = course.get("courseNumber")
    title = course.get("name")  # replaced with title
    description = course.get("description")
    min_credits = course.get("minCredits")
    max_credits = course.get("maxCredits")

    # Combine unique instructors from all sections
    instructors = set()
    for section in course.get("sections", []):
        faculty = section.get("faculty")
        if faculty:
            instructors.add(faculty)

    # Insert into database
    cur.execute("""
        INSERT INTO courses (
            subject, number, title, description,
            min_credits, max_credits, instructor
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        subject, int(number), title, description,
        min_credits, max_credits, list(instructors)
    ))

# === Commit and close ===
conn.commit()
cur.close()
conn.close()

print("✅ Data imported successfully!")
