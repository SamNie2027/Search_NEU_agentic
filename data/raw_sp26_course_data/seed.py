"""
Database seeding script for course data.

Loads course data from JSON file and inserts it into PostgreSQL database.
"""
import json
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

JSON_FILE = "data/raw_sp26_course_data/raw_data.json"

# === Connect to PostgreSQL ===
# Prefer DATABASE_URL (libpq/URI) if provided; otherwise fall back to a simple config.
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    conn = psycopg2.connect(DATABASE_URL)
else:
    # Fallback configuration (used previously)
    DB_CONFIG = {
        "dbname": "search_neu_agentic",
        "user": "postgres",
        "password": "12345678",
        "host": "localhost",
        "port": "5432",
    }
    conn = psycopg2.connect(**DB_CONFIG)

cur = conn.cursor()

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

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for course in data["courses"]:
    subject = course.get("subject")
    number = course.get("courseNumber")
    title = course.get("name")
    description = course.get("description")
    min_credits = course.get("minCredits")
    max_credits = course.get("maxCredits")

    instructors = set()
    for section in course.get("sections", []):
        faculty = section.get("faculty")
        if faculty:
            instructors.add(faculty)

    try:
        num_val = int(number) if number is not None and str(number).strip() != "" else None
    except (ValueError, TypeError):
        num_val = None

    cur.execute("""
        INSERT INTO courses (
            subject, number, title, description,
            min_credits, max_credits, instructors
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        subject, num_val, title, description,
        min_credits, max_credits, list(instructors)
    ))

conn.commit()
cur.close()
conn.close()

print("✅ Data imported successfully!")
