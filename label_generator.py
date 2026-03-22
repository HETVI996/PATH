"""
label_generator.py
──────────────────
Generates a Career_Track label for every student in the dataset using
a weighted scoring system across 7 career tracks.

HOW IT WORKS
------------
Each student's survey answers contribute points to each of 7 career tracks.
The track with the highest total score is assigned as their Career_Track.
This is more robust than hard if-else rules because:
  - Overlapping profiles get resolved by total evidence weight
  - No single answer can force an incorrect assignment
  - Ties are rare and resolved by the highest-scoring track

CAREER TRACKS
-------------
  1. Technology & Engineering
  2. Data & Research
  3. Creative & Design
  4. Business & Finance
  5. Healthcare & Wellness
  6. Social Impact & Education
  7. Management & Leadership

USAGE
-----
Run from the project root:
    python label_generator.py

Output: data/student_profiles_labeled.csv
Then update data_ingestion.py to read from that file.
"""

import os
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "student_astro_980_profiles_unique.csv"
OUTPUT_CSV = os.path.join("data", "student_profiles_labeled.csv")
TARGET_COL = "Career_Track"

TRACKS = [
    "Technology & Engineering",
    "Data & Research",
    "Creative & Design",
    "Business & Finance",
    "Healthcare & Wellness",
    "Social Impact & Education",
    "Management & Leadership",
]


# ── SCORING FUNCTION ──────────────────────────────────────────────────────────
def score_row(r) -> str:
    """
    Returns the best-fit career track for a single student row.
    Weights are calibrated so each track has at least 2–3 high-weight
    signals that strongly identify it, avoiding ties.
    """
    scores = {t: 0 for t in TRACKS}

    # ── Technology & Engineering ──────────────────────────────────────────────
    te = "Technology & Engineering"
    if r["Flow"] == "Coding":                                          scores[te] += 4
    if r["Curiosity"] == "Technology":                                 scores[te] += 4
    if r["Bookstore"] == "Science/Technology":                         scores[te] += 3
    if r["Math"] >= 7:                                                 scores[te] += 2
    if r["Friend_Help"] == "Technology":                               scores[te] += 2
    if r["Childhood"] == "Building":                                   scores[te] += 2
    if r["Hobbies"] in ("DIY Projects, Robotics", "Reading, Coding, Chess"):
                                                                       scores[te] += 3
    if r["Pride_Project"] in ("Built a personal website",
                               "Built an automation script",
                               "Created a mobile app",
                               "Won a hackathon"):                     scores[te] += 3
    if r["Energy"] in ("Solving complex problems energizes me",
                        "Building things keeps me engaged"):           scores[te] += 3
    if r["No_Money_Problem"] in ("Build sustainable technology",
                                  "Reduce digital divide"):            scores[te] += 2
    if r["Fulfillment"] == "Create":                                   scores[te] += 1
    if r["Environment"] == "Startup":                                  scores[te] += 1

    # ── Data & Research ───────────────────────────────────────────────────────
    dr = "Data & Research"
    if r["Group_Role"] == "Detective":                                 scores[dr] += 5
    if r["Flow"] == "Coding":                                          scores[dr] += 2
    if r["Math"] >= 7:                                                 scores[dr] += 3
    if r["Bookstore"] == "Science/Technology":                         scores[dr] += 2
    if r["Energy"] in ("Solving complex problems energizes me",
                        "Deep focused work energizes me",
                        "Learning new concepts motivates me"):         scores[dr] += 3
    if r["Pride_Project"] == "Completed a research mini-project":     scores[dr] += 4
    if r["Thinking"] == "Alone":                                       scores[dr] += 1
    if r["Hobbies"] in ("Reading, Coding, Chess",
                         "Podcast listening, Note-taking"):            scores[dr] += 2
    if r["Decision"] == "Head":                                        scores[dr] += 1
    if r["Childhood"] == "Exploring":                                  scores[dr] += 2
    if r["Work_Rhythm"] == "One big project":                          scores[dr] += 1

    # ── Creative & Design ─────────────────────────────────────────────────────
    cd = "Creative & Design"
    if r["Creativity"] >= 7:                                           scores[cd] += 4
    if r["Flow"] == "Writing":                                         scores[cd] += 4
    if r["Childhood"] == "Storytelling":                               scores[cd] += 3
    if r["Hobbies"] in ("Art, Sketching, Design",
                         "Writing, Blogging, Music",
                         "Gaming, Streaming, Editing",
                         "Travel, Photography, Journaling"):           scores[cd] += 4
    if r["Energy"] in ("Creative discussions excite me",
                        "Exploring new ideas excites me"):             scores[cd] += 3
    if r["Pride_Project"] in ("Published a blog series",
                               "Designed a prototype model"):          scores[cd] += 3
    if r["Bookstore"] == "Fiction":                                    scores[cd] += 3
    if r["Fulfillment"] == "Create":                                   scores[cd] += 2
    if r["Structure"] == "Freedom":                                    scores[cd] += 1
    if r["Curiosity"] == "Travel":                                     scores[cd] += 1

    # ── Business & Finance ────────────────────────────────────────────────────
    bf = "Business & Finance"
    if r["Curiosity"] == "Finance":                                    scores[bf] += 5
    if r["Hobbies"] == "Finance tracking, Investing":                  scores[bf] += 5
    if r["Energy"] == "Strategic planning excites me":                 scores[bf] += 3
    if r["Success"] == "Financial independence":                       scores[bf] += 3
    if r["No_Money_Problem"] in ("Improve financial literacy",
                                  "Support small businesses",
                                  "Reduce unemployment"):              scores[bf] += 3
    if r["Decision"] == "Head":                                        scores[bf] += 2
    if r["Management"] >= 7:                                           scores[bf] += 2
    if r["Bookstore"] == "Self-Help":                                  scores[bf] += 2
    if r["Environment"] == "Corporate":                                scores[bf] += 1

    # ── Healthcare & Wellness ─────────────────────────────────────────────────
    hw = "Healthcare & Wellness"
    if r["Curiosity"] == "Health":                                     scores[hw] += 5
    if r["No_Money_Problem"] in ("Solve healthcare affordability",
                                  "Improve mental health awareness"):  scores[hw] += 5
    if r["Energy"] == "Helping people gives me energy":                scores[hw] += 3
    if r["Hobbies"] == "Sports, Fitness, Yoga":                        scores[hw] += 3
    if r["Friend_Help"] == "Emotional Support":                        scores[hw] += 2
    if r["Fulfillment"] == "Help people":                              scores[hw] += 2
    if r["Decision"] == "Heart":                                       scores[hw] += 1
    if r["Environment"] == "Field-based":                              scores[hw] += 1

    # ── Social Impact & Education ─────────────────────────────────────────────
    si = "Social Impact & Education"
    if r["Energy"] == "Teaching others feels energizing":              scores[si] += 5
    if r["Hobbies"] == "Volunteering, Teaching":                       scores[si] += 5
    if r["No_Money_Problem"] in ("Improve education access",
                                  "Improve environmental sustainability",
                                  "Improve public safety systems"):    scores[si] += 4
    if r["Pride_Project"] in ("Organised a social drive",
                               "Led a college event"):                 scores[si] += 3
    if r["Success"] in ("Positive impact on society",
                         "Doing meaningful work",
                         "Helping others succeed"):                    scores[si] += 2
    if r["Fulfillment"] == "Help people":                              scores[si] += 2
    if r["Friend_Help"] == "Emotional Support":                        scores[si] += 2
    if r["Decision"] == "Heart":                                       scores[si] += 2
    if r["Bookstore"] == "History":                                    scores[si] += 1
    if r["Childhood"] == "Exploring":                                  scores[si] += 1
    if r["Curiosity"] == "Travel":                                     scores[si] += 1

    # ── Management & Leadership ───────────────────────────────────────────────
    ml = "Management & Leadership"
    if r["Group_Role"] == "Orchestrator":                              scores[ml] += 6
    if r["Management"] >= 7:                                           scores[ml] += 4
    if r["Childhood"] == "Organizing":                                 scores[ml] += 3
    if r["Friend_Help"] == "Organizing":                               scores[ml] += 3
    if r["Energy"] == "Strategic planning excites me":                 scores[ml] += 3
    if r["Work_Rhythm"] == "Many small tasks":                         scores[ml] += 2
    if r["Thinking"] == "Team":                                        scores[ml] += 2
    if r["Pride_Project"] == "Led a college event":                    scores[ml] += 2
    if r["Environment"] == "Corporate":                                scores[ml] += 1
    if r["Structure"] == "Structure":                                  scores[ml] += 1

    return max(scores, key=scores.get)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}")

    df[TARGET_COL] = df.apply(score_row, axis=1)

    print("\nCareer_Track distribution:")
    print(df[TARGET_COL].value_counts().to_string())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved labeled dataset to: {OUTPUT_CSV}")
    print("Next: update data_ingestion.py to read from this file.")


if __name__ == "__main__":
    main()