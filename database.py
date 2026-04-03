"""
database.py
───────────
SQLite database for PATH — stores every survey submission.

HOW IT IMPROVES THE MODEL OVER TIME
─────────────────────────────────────
Every user who completes the survey adds one row to path.db.
When you have enough new rows, run retrain_model.py — it combines
the original 17,180 training rows with all database rows and retrains.
More diverse, real-world data = better accuracy.

SUGGESTED RETRAIN MILESTONES
─────────────────────────────
  +200 rows  → small improvement, worth doing
  +500 rows  → noticeable improvement on weak classes
  +2000 rows → meaningful accuracy gain expected
  +5000 rows → model should consistently exceed 92%

TABLES
──────
responses — one row per submission, all 29 survey fields + predictions
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("data", "path.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Creates the responses table if it doesn't exist.
    Safe to call on every app startup — does nothing if table already exists.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at     TEXT NOT NULL,

            -- Structured questions (radio / likert)
            Bookstore        TEXT,
            Curiosity        TEXT,
            Flow             TEXT,
            Childhood        TEXT,
            Hobbies          TEXT,
            Friend_Help      TEXT,
            Math             INTEGER,
            Language         INTEGER,
            Creativity       INTEGER,
            Management       INTEGER,
            Group_Role       TEXT,
            Work_Rhythm      TEXT,
            Thinking         TEXT,
            Structure        TEXT,
            Decision         TEXT,
            Fulfillment      TEXT,
            Regret           TEXT,
            Environment      TEXT,

            -- 3 new domain columns (v2 dataset)
            Domain_Strength  TEXT,
            Work_Mode        TEXT,
            Output_Form      TEXT,

            -- Free-text questions (stored as typed by user)
            Pride_Project    TEXT,
            Energy           TEXT,
            Job_Choice       TEXT,
            No_Money_Problem TEXT,
            Success          TEXT,
            Career_Feel      TEXT,
            Ideal_Week       TEXT,
            Failure          TEXT,

            -- Prediction output
            predicted_track_1  TEXT,
            predicted_track_2  TEXT,
            predicted_track_3  TEXT,
            confidence_1       REAL,
            confidence_2       REAL,
            confidence_3       REAL
        )
    """)
    conn.commit()
    conn.close()


def save_response(raw_inputs: dict, classified_inputs: dict, predictions: list):
    """
    Saves a complete survey submission to the database.

    Parameters
    ----------
    raw_inputs        : dict — all form values from request.form
    classified_inputs : dict — same as raw_inputs for this pipeline
                        (kept for API compatibility with older code)
    predictions       : list of dicts from PredictPipeline.predict()
                        [{"rank":1,"track":"...","confidence":68.4}, ...]
    """
    conn = get_connection()
    cursor = conn.cursor()

    def get(key, default=""):
        val = raw_inputs.get(key, default)
        if isinstance(val, list):
            val = val[0] if val else default
        return val

    def get_int(key, default=5):
        try:
            return int(get(key, default))
        except (ValueError, TypeError):
            return default

    p1 = predictions[0] if len(predictions) > 0 else {}
    p2 = predictions[1] if len(predictions) > 1 else {}
    p3 = predictions[2] if len(predictions) > 2 else {}

    cursor.execute("""
        INSERT INTO responses (
            submitted_at,
            Bookstore, Curiosity, Flow, Childhood, Hobbies, Friend_Help,
            Math, Language, Creativity, Management,
            Group_Role, Work_Rhythm, Thinking, Structure, Decision,
            Fulfillment, Regret, Environment,
            Domain_Strength, Work_Mode, Output_Form,
            Pride_Project, Energy, Job_Choice, No_Money_Problem,
            Success, Career_Feel, Ideal_Week, Failure,
            predicted_track_1, predicted_track_2, predicted_track_3,
            confidence_1, confidence_2, confidence_3
        ) VALUES (
            ?,
            ?,?,?,?,?,?,
            ?,?,?,?,
            ?,?,?,?,?,
            ?,?,?,
            ?,?,?,
            ?,?,?,?,
            ?,?,?,?,
            ?,?,?,
            ?,?,?
        )
    """, (
        datetime.now().isoformat(),
        get("Bookstore"), get("Curiosity"), get("Flow"), get("Childhood"),
        get("Hobbies"), get("Friend_Help"),
        get_int("Math"), get_int("Language"), get_int("Creativity"), get_int("Management"),
        get("Group_Role"), get("Work_Rhythm"), get("Thinking"),
        get("Structure"), get("Decision"),
        get("Fulfillment"), get("Regret"), get("Environment"),
        get("Domain_Strength"), get("Work_Mode"), get("Output_Form"),
        get("Pride_Project"), get("Energy"), get("Job_Choice"),
        get("No_Money_Problem"), get("Success"), get("Career_Feel"),
        get("Ideal_Week"), get("Failure"),
        p1.get("track"), p2.get("track"), p3.get("track"),
        p1.get("confidence"), p2.get("confidence"), p3.get("confidence"),
    ))

    conn.commit()
    conn.close()


def get_all_responses_as_df():
    """
    Returns all stored responses as a pandas DataFrame,
    formatted to match the training data schema exactly.
    Used by retrain_model.py.
    """
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            Bookstore, Curiosity, Flow, Childhood,
            Pride_Project, Hobbies, Energy, Friend_Help,
            Math, Language, Creativity, Management,
            Group_Role, Work_Rhythm, Thinking, Structure,
            Decision, Job_Choice, No_Money_Problem, Fulfillment,
            Success, Career_Feel, Regret, Ideal_Week,
            Environment, Failure,
            Domain_Strength, Work_Mode, Output_Form,
            predicted_track_1 AS Career_Track
        FROM responses
        WHERE predicted_track_1 IS NOT NULL
    """, conn)
    conn.close()
    return df


def get_response_count():
    """Returns total number of responses stored."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    conn.close()
    return count


def get_track_counts():
    """Returns response count broken down by predicted track."""
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT predicted_track_1 as track, COUNT(*) as count
        FROM responses
        WHERE predicted_track_1 IS NOT NULL
        GROUP BY predicted_track_1
        ORDER BY count DESC
    """, conn)
    conn.close()
    return df


if __name__ == "__main__":
    init_db()
    count = get_response_count()
    print(f"Database initialised at: {DB_PATH}")
    print(f"Total responses stored:  {count}")
    if count > 0:
        print("\nResponses by track:")
        print(get_track_counts().to_string(index=False))