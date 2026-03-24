"""
label_generator.py
──────────────────
Generates Career_Track labels for the 17,180-row dataset using a
weighted scoring system across 13 career tracks.

CAREER TRACKS (13)
------------------
  1.  Technology & Engineering
  2.  Data & Research
  3.  Natural Sciences
  4.  Creative & Design
  5.  Fine Arts & Performing Arts
  6.  Interior Design & Architecture
  7.  Business & Finance
  8.  Healthcare & Wellness
  9.  Psychology & Counselling
  10. Social Impact & Education
  11. Policy, Law & Public Service
  12. Sports & Fitness
  13. Management & Leadership

USAGE
-----
    python label_generator.py

Output: data/student_profiles_labeled.csv
"""

import os
import pandas as pd

INPUT_CSV  = "student_astro_14780_final.csv"
OUTPUT_CSV = os.path.join("data", "student_profiles_labeled.csv")

TRACKS = [
    "Technology & Engineering",
    "Data & Research",
    "Natural Sciences",
    "Creative & Design",
    "Fine Arts & Performing Arts",
    "Interior Design & Architecture",
    "Business & Finance",
    "Healthcare & Wellness",
    "Psychology & Counselling",
    "Social Impact & Education",
    "Policy, Law & Public Service",
    "Sports & Fitness",
    "Management & Leadership",
]


def score_row(r) -> str:
    s = {t: 0 for t in TRACKS}
    pp = r["Pride_Project"].lower()
    en = r["Energy"].lower()

    # ── Technology & Engineering ──────────────────────────────────────
    te = "Technology & Engineering"
    if r["Curiosity"] == "Technology":                                 s[te] += 6
    if r["Flow"] == "Coding":                                          s[te] += 6
    if r["Environment"] == "Startup":                                  s[te] += 3
    if r["Friend_Help"] == "Technology":                               s[te] += 2
    if r["Childhood"] == "Building":                                   s[te] += 2
    if r["Bookstore"] == "Science/Technology":                         s[te] += 2
    if r["Hobbies"] in ("DIY Projects, Robotics", "Reading, Coding, Chess"): s[te] += 3
    if r["Math"] >= 7:                                                 s[te] += 2
    if any(k in pp for k in ["web app","mobile app","automation","hackathon",
                               "full-stack","websocket","machine learning",
                               "raspberry pi","chatbot","open-source","data pipeline",
                               "neural","algorithm","deployment"]):   s[te] += 4

    # ── Natural Sciences ──────────────────────────────────────────────
    ns = "Natural Sciences"
    if r["Curiosity"] == "Science":                                    s[ns] += 7
    if r["Flow"] == "Experiments":                                     s[ns] += 7
    if r["Environment"] == "Research lab":                             s[ns] += 6
    if r["Hobbies"] == "Science experiments, Nature observation":      s[ns] += 5
    if r["Math"] >= 7:                                                 s[ns] += 2
    if any(k in pp for k in ["science olympiad","molecular biology","chemistry",
                               "biology","csir","antibiotic","physics",
                               "national science","genome","soil microbiome",
                               "water purification","published a paper on"]):  s[ns] += 5

    # ── Data & Research ───────────────────────────────────────────────
    dr = "Data & Research"
    if r["Group_Role"] == "Detective":                                 s[dr] += 5
    if r["Flow"] == "Experiments":                                     s[dr] += 2
    if r["Environment"] == "Research lab":                             s[dr] += 2
    if r["Thinking"] == "Alone":                                       s[dr] += 1
    if r["Math"] >= 7:                                                 s[dr] += 3
    if r["Hobbies"] in ("Reading, Coding, Chess",
                         "Podcast listening, Note-taking",
                         "Science experiments, Nature observation"):   s[dr] += 2
    if any(k in pp for k in ["research mini-project","data pipeline",
                               "image classification","independent study",
                               "analysis","github","open-source"]):   s[dr] += 4

    # ── Fine Arts & Performing Arts ───────────────────────────────────
    fa = "Fine Arts & Performing Arts"
    if r["Curiosity"] == "Arts / Culture":                             s[fa] += 8
    if r["Flow"] == "Drawing / Making":                                s[fa] += 7
    if r["Environment"] == "Studio / Gallery":                         s[fa] += 5
    if r["Hobbies"] == "Painting, Sculpting, Music":                   s[fa] += 5
    if r["Creativity"] >= 8:                                           s[fa] += 3
    if any(k in pp for k in ["residency","fine arts","exhibited artwork",
                               "painting","sculpture","performed","gallery",
                               "music ep","composed","directed","short film",
                               "theatre","mural"]):                    s[fa] += 5

    # ── Interior Design & Architecture ───────────────────────────────
    id_ = "Interior Design & Architecture"
    if any(k in pp for k in ["interior design","floor plan","autocad",
                               "co-working space","mood board","material palette",
                               "boutique interior","sustainable architecture",
                               "upcycled","cafe concept","stage and backdrop",
                               "residential and retail","design studio"]):  s[id_] += 10
    if r["Creativity"] >= 8:                                           s[id_] += 2
    if r["Environment"] == "Studio / Gallery":                         s[id_] += 2

    # ── Creative & Design ─────────────────────────────────────────────
    cd = "Creative & Design"
    if r["Childhood"] == "Storytelling":                               s[cd] += 3
    if r["Bookstore"] == "Fiction":                                    s[cd] += 2
    if r["Hobbies"] in ("Writing, Blogging, Music",
                         "Travel, Photography, Journaling",
                         "Art, Sketching, Design",
                         "Gaming, Streaming, Editing"):                s[cd] += 3
    # Only add creativity bonus if not already dominated by Fine Arts or Interior Design
    if r["Creativity"] >= 7 and s[fa] < 3 and s[id_] < 3:            s[cd] += 3
    if any(k in pp for k in ["blog series","graphic novel","photo essay",
                               "storytelling workshop","short film competition"]):  s[cd] += 4

    # ── Business & Finance ────────────────────────────────────────────
    bf = "Business & Finance"
    if r["Curiosity"] == "Finance":                                    s[bf] += 7
    if r["Hobbies"] == "Finance tracking, Investing":                  s[bf] += 6
    if r["Environment"] == "Corporate":                                s[bf] += 2
    if r["Bookstore"] == "Self-Help":                                  s[bf] += 2
    if r["Management"] >= 7:                                           s[bf] += 2
    if any(k in pp for k in ["b-school","financial model","valuation",
                               "pe or vc","consulting firm","business plan",
                               "seed funding","entrepreneurship summit",
                               "cost-saving","social enterprise"]):    s[bf] += 6

    # ── Healthcare & Wellness ─────────────────────────────────────────
    hw = "Healthcare & Wellness"
    if r["Curiosity"] == "Health":                                     s[hw] += 3
    if r["Environment"] == "Field-based":                              s[hw] += 2
    if r["Fulfillment"] == "Help people":                              s[hw] += 1
    if any(k in pp for k in ["hospital","icu","health camp","immunisation",
                               "palliative","first-aid","blood donation",
                               "cancer care","trauma care","patient recovery",
                               "ward rounds","clinical","public health"]):  s[hw] += 8

    # ── Psychology & Counselling ──────────────────────────────────────
    ps = "Psychology & Counselling"
    if any(k in pp for k in ["counselling","psychology","therapy",
                               "mental health awareness","cbt","peer counsellor",
                               "wellbeing survey","stress and coping","social anxiety",
                               "attachment styles","group therapy","practicum",
                               "rehabilitation","trafficking survivor"]):  s[ps] += 9
    if any(k in en for k in ["emotional intelligence","human behaviour",
                               "psychology","counselling","mental health"]):  s[ps] += 4
    if r["Curiosity"] == "Health" and s[ps] > 0:                      s[ps] += 3
    if r["Decision"] == "Heart":                                       s[ps] += 1
    if r["Fulfillment"] == "Help people":                              s[ps] += 1

    # ── Social Impact & Education ─────────────────────────────────────
    si = "Social Impact & Education"
    if r["Hobbies"] == "Volunteering, Teaching":                       s[si] += 5
    if r["Fulfillment"] == "Help people":                              s[si] += 2
    if r["Bookstore"] == "History":                                    s[si] += 2
    if r["Decision"] == "Heart":                                       s[si] += 1
    if any(k in pp for k in ["tuition programme","volunteer fellowship",
                               "rural children","b.ed","social drive",
                               "community initiative","underprivileged",
                               "social welfare","street children","sanitation",
                               "voter awareness","msw","child welfare",
                               "livelihood training"]):                s[si] += 7
    if any(k in en for k in ["teaching others feels energizing","educating"]): s[si] += 4

    # ── Policy, Law & Public Service ─────────────────────────────────
    pl = "Policy, Law & Public Service"
    if r["Curiosity"] == "Justice / Law":                              s[pl] += 9
    if r["Curiosity"] == "Politics / Policy":                          s[pl] += 8
    if r["Flow"] == "Debating":                                        s[pl] += 7
    if r["Environment"] in ("Government office", "Courtroom / Chambers"): s[pl] += 7
    if r["Hobbies"] in ("Debating / MUNs", "Current affairs, Reading"): s[pl] += 5
    if any(k in pp for k in ["law fest","legal awareness","supreme court",
                               "juvenile justice","mun","constitutional",
                               "district collector","law school","access to justice",
                               "debate competition","data privacy law",
                               "public policy"]):                      s[pl] += 5

    # ── Sports & Fitness ─────────────────────────────────────────────
    sf = "Sports & Fitness"
    if r["Curiosity"] == "Sports / Fitness":                           s[sf] += 9
    if r["Environment"] == "Outdoor / Training ground":                s[sf] += 8
    if r["Hobbies"] == "Sports, Fitness, Yoga":                        s[sf] += 3
    if any(k in pp for k in ["athletics","cricket","football","swimming",
                               "kabaddi","sprint","trained and qualified",
                               "captained","fitness training",
                               "national-level","state-level",
                               "sports meet"]):                        s[sf] += 6

    # ── Management & Leadership ───────────────────────────────────────
    ml = "Management & Leadership"
    if r["Group_Role"] == "Orchestrator":                              s[ml] += 6
    if r["Management"] >= 7:                                           s[ml] += 4
    if r["Childhood"] == "Organizing":                                 s[ml] += 3
    if r["Friend_Help"] == "Organizing":                               s[ml] += 3
    if r["Work_Rhythm"] == "Many small tasks":                         s[ml] += 2
    if r["Thinking"] == "Team":                                        s[ml] += 2
    if any(k in pp for k in ["managed","led a 15-member","fest budget",
                               "entrepreneurship summit","student council"]): s[ml] += 4

    return max(s, key=s.get)


def main():
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Rows: {len(df)}")

    df["Career_Track"] = df.apply(score_row, axis=1)

    print("\nCareer_Track distribution:")
    print(df["Career_Track"].value_counts().to_string())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()