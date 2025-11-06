import json
import os
import psycopg2
from dotenv import load_dotenv

# Load .env if present so DATABASE_URL can be provided from workspace settings
load_dotenv()

JSON_FILE = "raw_data/raw_data.json"

# === Connect to PostgreSQL ===
# Prefer DATABASE_URL (libpq/URI) if provided; otherwise fall back to a simple config.
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # psycopg2 accepts a libpq connection string / URI
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

    # Convert number safely to int when possible (DB expects INTEGER)
    try:
        num_val = int(number) if number is not None and str(number).strip() != "" else None
    except (ValueError, TypeError):
        num_val = None

    # Insert into database
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

# === Commit and close ===
conn.commit()
cur.close()
conn.close()

print("✅ Data imported successfully!")
