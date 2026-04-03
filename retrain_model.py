"""
retrain_model.py
────────────────
Combines the original 17,180-row training dataset with all new responses
stored in the database, then retrains the model and saves fresh artifacts.

WHEN TO RUN
───────────
Run this script whenever you want to improve accuracy with new data.

    python retrain_model.py

Suggested milestones:
  +200  DB rows  → first retrain, expect ~0.5–1% improvement
  +500  DB rows  → improvement on weaker classes
  +2000 DB rows  → consistent accuracy gain
  +5000 DB rows  → model should exceed 92%

Check how many responses you have at any time:
    python -c "from database import get_response_count; print(get_response_count())"

HOW IT WORKS
────────────
1. Loads the original labeled CSV  (data/student_profiles_labeled.csv)
2. Loads all DB responses          (data/path.db)
3. Re-applies the scoring function to DB responses to assign Career_Track
4. Combines both datasets
5. Retrains Logistic Regression with the same pipeline
6. Saves new artifacts/ — app.py picks them up on next restart
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import get_all_responses_as_df, get_response_count
from src.utils import save_object


TRACKS = [
    "Technology & Engineering", "Data & Research", "Natural Sciences",
    "Creative & Design", "Fine Arts & Performing Arts",
    "Interior Design & Architecture", "Business & Finance",
    "Healthcare & Wellness", "Psychology & Counselling",
    "Social Impact & Education", "Policy, Law & Public Service",
    "Sports & Fitness", "Management & Leadership",
]

STRUCTURED_COLS = [
    "Bookstore", "Curiosity", "Flow", "Childhood", "Friend_Help",
    "Group_Role", "Work_Rhythm", "Thinking", "Structure", "Decision",
    "Fulfillment", "Regret", "Environment", "Hobbies",
    "Domain_Strength", "Work_Mode", "Output_Form",
]
NUMERIC_COLS = ["Math", "Language", "Creativity", "Management"]
TFIDF_CONFIGS = {
    "Pride_Project":    dict(max_features=2000, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    "Energy":           dict(max_features=800,  ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    "Job_Choice":       dict(max_features=400,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
    "No_Money_Problem": dict(max_features=400,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
    "Success":          dict(max_features=300,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
    "Ideal_Week":       dict(max_features=300,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
    "Failure":          dict(max_features=300,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
}


def score_row(r) -> str:
    """Same scoring as label_generator.py — assigns Career_Track to DB responses."""
    s = {t: 0 for t in TRACKS}
    pp = str(r.get("Pride_Project", "")).lower()
    en = str(r.get("Energy", "")).lower()

    te = "Technology & Engineering"
    if r.get("Curiosity") == "Technology":           s[te] += 6
    if r.get("Flow") == "Coding":                    s[te] += 6
    if r.get("Environment") == "Startup":            s[te] += 3
    if r.get("Friend_Help") == "Technology":         s[te] += 2
    if r.get("Childhood") == "Building":             s[te] += 2
    if r.get("Hobbies") in ("DIY Projects, Robotics","Reading, Coding, Chess"): s[te] += 3
    if any(k in pp for k in ["web app","mobile app","automation","hackathon","full-stack",
                               "machine learning","raspberry pi","chatbot","open-source","data pipeline"]): s[te] += 4
    if (r.get("Math") or 0) >= 7:                   s[te] += 2
    if r.get("Domain_Strength") == "STEM" and r.get("Work_Mode") == "Technical/Build": s[te] += 4

    ns = "Natural Sciences"
    if r.get("Curiosity") == "Science":              s[ns] += 7
    if r.get("Flow") == "Experiments":               s[ns] += 7
    if r.get("Environment") == "Research lab":       s[ns] += 6
    if r.get("Hobbies") == "Science experiments, Nature observation": s[ns] += 5
    if any(k in pp for k in ["science olympiad","molecular biology","chemistry","biology",
                               "csir","antibiotic","physics","national science","genome"]): s[ns] += 5
    if r.get("Domain_Strength") == "STEM" and r.get("Work_Mode") == "Lab/Research": s[ns] += 5

    dr = "Data & Research"
    if r.get("Group_Role") == "Detective":           s[dr] += 6
    if r.get("Flow") == "Experiments":               s[dr] += 2
    if r.get("Environment") == "Research lab":       s[dr] += 2
    if r.get("Thinking") == "Alone":                 s[dr] += 2
    math_val = r.get("Math") or 0
    if math_val >= 8:                                s[dr] += 4
    elif math_val >= 7:                              s[dr] += 2
    if (r.get("Language") or 10) <= 6:              s[dr] += 1
    if any(k in pp for k in ["research mini-project","data pipeline","image classification",
                               "independent study","analysis","github"]): s[dr] += 4
    if r.get("Output_Form") == "Data/Analysis":      s[dr] += 4

    fa = "Fine Arts & Performing Arts"
    if r.get("Curiosity") == "Arts / Culture":       s[fa] += 8
    if r.get("Flow") == "Drawing / Making":          s[fa] += 7
    if r.get("Environment") == "Studio / Gallery":   s[fa] += 5
    if r.get("Hobbies") == "Painting, Sculpting, Music": s[fa] += 5
    if (r.get("Creativity") or 0) >= 8:              s[fa] += 3
    if any(k in pp for k in ["residency","fine arts","exhibited artwork","painting","sculpture",
                               "performed","gallery","music ep","composed","directed",
                               "short film","theatre","mural"]): s[fa] += 5
    if r.get("Domain_Strength") == "Arts" and r.get("Work_Mode") == "Creative/Communication": s[fa] += 5

    id_ = "Interior Design & Architecture"
    if any(k in pp for k in ["interior design","floor plan","autocad","co-working space",
                               "mood board","material palette","boutique interior",
                               "sustainable architecture","upcycled","cafe concept",
                               "stage and backdrop"]): s[id_] += 10
    if (r.get("Creativity") or 0) >= 8:              s[id_] += 2
    if r.get("Environment") == "Studio / Gallery":   s[id_] += 2
    if r.get("Domain_Strength") == "Arts" and r.get("Work_Mode") == "Creative/Communication": s[id_] += 3

    cd = "Creative & Design"
    if r.get("Childhood") == "Storytelling":         s[cd] += 4
    if r.get("Flow") == "Writing":                   s[cd] += 5
    if r.get("Flow") == "Reading / Writing":         s[cd] += 2
    if r.get("Hobbies") in ("Writing, Blogging, Music","Travel, Photography, Journaling"): s[cd] += 4
    if r.get("Bookstore") == "Fiction":              s[cd] += 2
    if (r.get("Language") or 0) >= 8:               s[cd] += 2
    if (r.get("Creativity") or 0) >= 7 and s[fa] < 3 and s[id_] < 3: s[cd] += 2
    if any(k in pp for k in ["blog series","graphic novel","photo essay",
                               "storytelling workshop","published","wrote","content"]): s[cd] += 4
    if r.get("Output_Form") == "Written/Published":  s[cd] += 3
    if r.get("Domain_Strength") == "Arts" and r.get("Work_Mode") == "Creative/Communication": s[cd] += 3

    bf = "Business & Finance"
    if r.get("Curiosity") == "Finance":              s[bf] += 7
    if r.get("Hobbies") == "Finance tracking, Investing": s[bf] += 6
    if r.get("Environment") == "Corporate":          s[bf] += 2
    if r.get("Bookstore") == "Self-Help":            s[bf] += 2
    if (r.get("Management") or 0) >= 7:              s[bf] += 2
    if any(k in pp for k in ["b-school","financial model","valuation","pe or vc",
                               "consulting firm","business plan","seed funding",
                               "entrepreneurship summit"]): s[bf] += 6
    if r.get("Domain_Strength") == "Business" and r.get("Work_Mode") == "Operations/Strategy": s[bf] += 5

    hw = "Healthcare & Wellness"
    if r.get("Curiosity") == "Health":               s[hw] += 3
    if r.get("Environment") == "Field-based":        s[hw] += 2
    if any(k in pp for k in ["hospital","icu","health camp","immunisation","palliative",
                               "first-aid","blood donation","cancer care","trauma care",
                               "patient recovery","ward rounds"]): s[hw] += 8
    if r.get("Domain_Strength") == "Clinical" and r.get("Work_Mode") == "Hands-on Clinical": s[hw] += 6
    if r.get("Output_Form") == "Physical/Patient":   s[hw] += 3

    ps = "Psychology & Counselling"
    if any(k in pp for k in ["counselling","psychology","therapy","mental health awareness",
                               "cbt","peer counsellor","wellbeing survey","stress and coping",
                               "social anxiety","attachment styles","group therapy","practicum",
                               "rehabilitation","trafficking survivor"]): s[ps] += 9
    if any(k in en for k in ["emotional intelligence","human behaviour","psychology",
                               "counselling","mental health"]): s[ps] += 4
    if r.get("Curiosity") == "Health" and s[ps] > 0: s[ps] += 3
    if r.get("Domain_Strength") == "Clinical" and r.get("Work_Mode") == "Hands-on Clinical": s[ps] += 4
    if r.get("Domain_Strength") == "Social" and r.get("Work_Mode") == "People/Community": s[ps] += 3

    si = "Social Impact & Education"
    if r.get("Hobbies") == "Volunteering, Teaching": s[si] += 5
    if r.get("Fulfillment") == "Help people":        s[si] += 2
    if r.get("Bookstore") == "History":              s[si] += 2
    if any(k in pp for k in ["tuition programme","volunteer fellowship","rural children",
                               "b.ed","social drive","community initiative","underprivileged",
                               "social welfare","street children","sanitation","voter awareness",
                               "msw","child welfare","livelihood training"]): s[si] += 7
    if any(k in en for k in ["teaching others feels energizing","educating"]): s[si] += 4
    if r.get("Domain_Strength") == "Social" and r.get("Work_Mode") == "Fieldwork/Community": s[si] += 5
    if r.get("Output_Form") == "Service/Experience": s[si] += 2

    pl = "Policy, Law & Public Service"
    if r.get("Curiosity") == "Justice / Law":        s[pl] += 9
    if r.get("Curiosity") == "Politics / Policy":    s[pl] += 8
    if r.get("Flow") == "Debating":                  s[pl] += 7
    if r.get("Environment") in ("Government office","Courtroom / Chambers"): s[pl] += 7
    if r.get("Hobbies") in ("Debating / MUNs","Current affairs, Reading"): s[pl] += 5
    if any(k in pp for k in ["law fest","legal awareness","supreme court","juvenile justice",
                               "mun","constitutional","district collector","law school",
                               "access to justice","debate competition","data privacy law"]): s[pl] += 5
    if r.get("Domain_Strength") == "Humanities/Law": s[pl] += 6
    if r.get("Output_Form") == "Written/Published" and r.get("Domain_Strength") == "Humanities/Law": s[pl] += 3

    sf = "Sports & Fitness"
    if r.get("Curiosity") == "Sports / Fitness":     s[sf] += 9
    if r.get("Environment") == "Outdoor / Training ground": s[sf] += 8
    if r.get("Hobbies") == "Sports, Fitness, Yoga":  s[sf] += 3
    if any(k in pp for k in ["athletics","cricket","football","swimming","kabaddi","sprint",
                               "trained and qualified","captained","fitness training",
                               "national-level","state-level"]): s[sf] += 6

    ml = "Management & Leadership"
    if r.get("Group_Role") == "Orchestrator":        s[ml] += 7
    mgmt = r.get("Management") or 0
    if mgmt >= 8:                                    s[ml] += 5
    elif mgmt >= 7:                                  s[ml] += 3
    if r.get("Childhood") == "Organizing":           s[ml] += 3
    if r.get("Friend_Help") == "Organizing":         s[ml] += 3
    if r.get("Work_Rhythm") == "Many small tasks":   s[ml] += 1
    if r.get("Thinking") == "Team":                  s[ml] += 1
    if any(k in pp for k in ["managed","led a 15-member","fest budget",
                               "entrepreneurship summit","student council"]): s[ml] += 5
    if r.get("Domain_Strength") == "Business" and r.get("Work_Mode") == "Operations/Strategy": s[ml] += 3

    return max(s, key=s.get)


def build_preprocessor():
    transformers = [
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), NUMERIC_COLS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), STRUCTURED_COLS),
    ]
    for col, cfg in TFIDF_CONFIGS.items():
        transformers.append((col.lower(), TfidfVectorizer(**cfg), col))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def load_combined_dataset():
    print("Loading original labeled dataset...")
    original = pd.read_csv(os.path.join("data", "student_profiles_labeled.csv"))
    print(f"  Original: {len(original)} rows")

    db_count = get_response_count()
    print(f"  Database responses: {db_count}")

    if db_count > 0:
        db_df = get_all_responses_as_df()

        # Re-score DB responses using the same scoring function
        # (predicted_track_1 is the label stored, but re-scoring adds consistency)
        print("  Scoring DB responses with label generator...")
        db_df["Career_Track"] = db_df.apply(score_row, axis=1)

        # Align columns to original schema
        feature_cols = [c for c in original.columns if c in db_df.columns]
        db_aligned = db_df[feature_cols]

        combined = pd.concat([original, db_aligned], ignore_index=True)
        print(f"  Combined total: {len(combined)} rows")
    else:
        combined = original
        print("  No DB responses yet — using original dataset only")

    print("\nCareer_Track distribution:")
    print(combined["Career_Track"].value_counts().to_string())
    print()
    return combined


def retrain():
    combined = load_combined_dataset()

    drop_cols = ["Student_ID", "Career_Feel", "Career_Track"]
    X = combined.drop(columns=[c for c in drop_cols if c in combined.columns])
    y = combined["Career_Track"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"Classes ({len(le.classes_)}): {le.classes_.tolist()}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("Building preprocessor and training model...")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    model = LogisticRegression(
        C=10.0, max_iter=3000, class_weight="balanced",
        solver="saga", n_jobs=-1
    )
    model.fit(X_train_t, y_train)

    y_pred   = model.predict(X_test_t)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, model.predict(X_train_t))

    proba = model.predict_proba(X_test_t)
    top3_acc = sum(
        1 for i, true in enumerate(y_test)
        if true in proba[i].argsort()[-3:][::-1]
    ) / len(y_test)

    print(f"\n{'='*52}")
    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Test  accuracy : {test_acc:.4f}")
    print(f"  Top-3 accuracy : {top3_acc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    save_object("artifacts/model.pkl",         model)
    save_object("artifacts/preprocessor.pkl",  preprocessor)
    save_object("artifacts/label_encoder.pkl", le)
    save_object("artifacts/class_names.pkl",   le.classes_.tolist())

    print("Artifacts saved:")
    print("  artifacts/model.pkl")
    print("  artifacts/preprocessor.pkl")
    print("  artifacts/label_encoder.pkl")
    print("  artifacts/class_names.pkl")
    print()
    print(f"Retrain complete. New test accuracy = {test_acc:.4f}")
    print("Restart python app.py to use the new model.")
    return test_acc


if __name__ == "__main__":
    retrain()