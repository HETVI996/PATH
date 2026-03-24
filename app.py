import sys
from flask import Flask, render_template, request, redirect, url_for

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)


# ── Survey question definitions ───────────────────────────────────────────────
QUESTIONS = [
    {
        "id":      "Bookstore",
        "number":  1,
        "title":   "The bookstore test",
        "question": "If you walked into a bookstore with unlimited time and money, where do you go first?",
        "type":    "radio",
        "options": ["Fiction", "Science/Technology", "History", "Self-Help"],
    },
    {
        "id":      "Curiosity",
        "number":  2,
        "title":   "Your curiosity",
        "question": "Outside work or school, what topics do you find yourself Googling or watching videos about just for fun?",
        "type":    "radio",
        "options": ["Technology", "Health", "Travel", "Finance"],
    },
    {
        "id":      "Flow",
        "number":  3,
        "title":   "Flow state",
        "question": "Which activity makes you completely lose track of time?",
        "type":    "radio",
        "options": ["Writing", "Coding", "Sports", "Games"],
    },
    {
        "id":      "Childhood",
        "number":  4,
        "title":   "Earliest interests",
        "question": "Looking back at childhood, what did you love doing most?",
        "type":    "radio",
        "options": ["Building", "Storytelling", "Organizing", "Exploring"],
    },
    {
        "id":      "Pride_Project",
        "number":  5,
        "title":   "Pride project",
        "question": "Think about a project or accomplishment you are genuinely proud of. Describe it briefly — what did you build, organise, research, or achieve?",
        "type":    "freetext",
        "hint":    "e.g. Built a hospital management app, Won a hackathon, Organised a blood donation drive…",
        "placeholder": "Describe your proudest project or achievement…",
    },
    {
        "id":      "Hobbies",
        "number":  6,
        "title":   "Your hobbies",
        "question": "What are at least 3 hobbies that you genuinely enjoy in your free time?",
        "type":    "freetext",
        "hint":    "e.g. Cricket, gym, reading about finance, sketching, coding side projects…",
        "placeholder": "List your hobbies…",
    },
    {
        "id":      "Energy",
        "number":  7,
        "title":   "Energy barometer",
        "question": "What conversations, tasks or subjects leave you feeling energised or excited? What drains you?",
        "type":    "freetext",
        "hint":    "e.g. Solving hard problems energises me. Repetitive admin work drains me.",
        "placeholder": "Describe what energises and drains you…",
    },
    {
        "id":      "Friend_Help",
        "number":  8,
        "title":   "Friend-sourced skill",
        "question": "What do your friends or family consistently ask you for help with?",
        "type":    "radio",
        "options": ["Technology", "Organizing", "Emotional Support", "Problem-Solving"],
    },
    {
        "id":      "Math",
        "number":  9,
        "title":   "Natural ease — mathematics",
        "question": "How intuitively easy did mathematics feel for you? (1 = very hard, 10 = effortless)",
        "type":    "likert",
        "min": 1, "max": 10,
    },
    {
        "id":      "Language",
        "number":  10,
        "title":   "Natural ease — language",
        "question": "How intuitively easy did language / writing feel for you? (1 = very hard, 10 = effortless)",
        "type":    "likert",
        "min": 1, "max": 10,
    },
    {
        "id":      "Creativity",
        "number":  11,
        "title":   "Natural ease — creativity",
        "question": "How naturally creative do you feel? (1 = not at all, 10 = extremely)",
        "type":    "likert",
        "min": 1, "max": 10,
    },
    {
        "id":      "Management",
        "number":  12,
        "title":   "Natural ease — management",
        "question": "How naturally do you take charge of organising people and tasks? (1 = never, 10 = always)",
        "type":    "likert",
        "min": 1, "max": 10,
    },
    {
        "id":      "Group_Role",
        "number":  13,
        "title":   "Problem-solving role",
        "question": "When working in a group on a difficult problem, what role do you naturally play?",
        "type":    "radio",
        "options": [
            "Idea generator — brainstorming creative, out-of-box ideas",
            "Orchestrator — organising the plan and delegating tasks",
            "Detective — researching information and finding the facts",
            "Builder — executing the plan and creating the final solution",
        ],
        "values": ["Idea generator", "Orchestrator", "Detective", "Builder"],
    },
    {
        "id":      "Work_Rhythm",
        "number":  14,
        "title":   "Work rhythm",
        "question": "How do you prefer to work?",
        "type":    "radio",
        "options": [
            "Deeply focus on one big project for a long period",
            "Enjoy the variety of juggling many smaller tasks at once",
        ],
        "values": ["One big project", "Many small tasks"],
    },
    {
        "id":      "Thinking",
        "number":  15,
        "title":   "Thinking style",
        "question": "When do you do your best thinking?",
        "type":    "radio",
        "options": [
            "Brainstorming and collaborating live with the team",
            "Working alone in a quiet space to think deeply",
            "A mix — collaborating for ideas, then working solo to execute",
        ],
        "values": ["Team", "Alone", "Mix"],
    },
    {
        "id":      "Structure",
        "number":  16,
        "title":   "Structure vs freedom",
        "question": "Which work environment sounds more appealing?",
        "type":    "radio",
        "options": [
            "A job with clear, predictable routines and well-defined tasks",
            "One with a flexible schedule and freedom to experiment",
        ],
        "values": ["Structure", "Freedom"],
    },
    {
        "id":      "Decision",
        "number":  17,
        "title":   "Decision-making style",
        "question": "When making an important decision, do you primarily rely on…",
        "type":    "radio",
        "options": [
            "Head — objective data, logic, and pros/cons",
            "Heart — intuition, personal values, and impact on people",
        ],
        "values": ["Head", "Heart"],
    },
    {
        "id":      "Job_Choice",
        "number":  18,
        "title":   "The non-negotiable test",
        "question": "Imagine two job offers — Job A has a significantly higher salary, Job B has something else you deeply value. Which would you choose and why?",
        "type":    "freetext",
        "hint":    "e.g. I'd choose B because work-life balance matters more to me than money right now.",
        "placeholder": "Describe your choice and reasoning…",
    },
    {
        "id":      "No_Money_Problem",
        "number":  19,
        "title":   "The no-money test",
        "question": "If money were no object, what problem in the world would you dedicate your life to solving?",
        "type":    "freetext",
        "hint":    "e.g. I would make quality mental healthcare free for everyone.",
        "placeholder": "Describe the problem you'd solve…",
    },
    {
        "id":      "Fulfillment",
        "number":  20,
        "title":   "Contribution vs creation",
        "question": "Which feels more fulfilling to you?",
        "type":    "radio",
        "options": [
            "Directly help, teach or serve people",
            "Create something new that people can use or experience",
        ],
        "values": ["Help people", "Create"],
    },
    {
        "id":      "Success",
        "number":  21,
        "title":   "Defining success",
        "question": "At the end of your life, what needs to be true for it to feel like a success?",
        "type":    "freetext",
        "hint":    "e.g. Having built something that changed how people live, or raised a happy family.",
        "placeholder": "Describe what success means to you…",
    },
    {
        "id":      "Career_Feel",
        "number":  22,
        "title":   "The party test",
        "question": "What one feeling do you want people to associate with your career story in 20 years?",
        "type":    "freetext",
        "hint":    "e.g. Innovation, Compassion, Leadership, Trust, Impact…",
        "placeholder": "One word or feeling…",
    },
    {
        "id":      "Regret",
        "number":  22,
        "title":   "Regret minimisation",
        "question": "Which long-term regret would you rather avoid?",
        "type":    "radio",
        "options": [
            "The regret of not taking a major risk for higher reward",
            "The regret of not choosing a path that guaranteed stability",
        ],
        "values": ["Risk", "Stability"],
    },
    {
        "id":      "Ideal_Week",
        "number":  23,
        "title":   "Ideal work week",
        "question": "Design your ideal work week. What does it look like day to day?",
        "type":    "freetext",
        "hint":    "e.g. Deep coding in the mornings, short team standup, evenings free for learning.",
        "placeholder": "Describe your ideal work week…",
    },
    {
        "id":      "Environment",
        "number":  24,
        "title":   "Work environment",
        "question": "Select your optimal work environment:",
        "type":    "radio",
        "options": ["Corporate", "Field-based", "Solo", "Startup"],
    },
    {
        "id":      "Failure",
        "number":  25,
        "title":   "Defining failure",
        "question": "What would a significant failure look like in your career, and how would you recover from it?",
        "type":    "freetext",
        "hint":    "e.g. Working for 5 years without growing. I'd recover by seeking mentorship and switching paths.",
        "placeholder": "Describe what failure looks like and how you'd recover…",
    },
]

# Career track descriptions shown on the results page
TRACK_INFO = {
    "Technology & Engineering": {
        "emoji": "⚙️",
        "desc":  "You thrive building systems, writing code, and solving technical challenges. Roles like software engineer, product developer, or systems architect align with your profile.",
        "roles": ["Software Engineer", "Product Developer", "DevOps Engineer", "Systems Architect"],
    },
    "Data & Research": {
        "emoji": "🔍",
        "desc":  "You have a detective's curiosity — you love digging into data, finding patterns, and turning insights into decisions. Roles in analytics, research, or data science suit you.",
        "roles": ["Data Scientist", "Research Analyst", "Business Intelligence Analyst", "ML Engineer"],
    },
    "Creative & Design": {
        "emoji": "🎨",
        "desc":  "You see the world as a canvas. Writing, design, storytelling and creative problem-solving energise you. Roles in UX, content, branding or media are a natural fit.",
        "roles": ["UX/UI Designer", "Content Strategist", "Brand Designer", "Creative Director"],
    },
    "Business & Finance": {
        "emoji": "📈",
        "desc":  "You think in numbers, strategies, and outcomes. Finance, consulting, or entrepreneurship let you build and scale things that matter economically.",
        "roles": ["Financial Analyst", "Business Consultant", "Entrepreneur", "Investment Analyst"],
    },
    "Healthcare & Wellness": {
        "emoji": "🩺",
        "desc":  "Your drive to help people thrive makes you a natural in healthcare, mental wellness, or public health. You combine empathy with impact.",
        "roles": ["Healthcare Professional", "Public Health Specialist", "Mental Health Counsellor", "Nutritionist"],
    },
    "Social Impact & Education": {
        "emoji": "🌍",
        "desc":  "You want to change systems, teach, and lift communities. NGOs, education, policy, and social enterprise are spaces where your values and skills converge.",
        "roles": ["Educator", "NGO Program Manager", "Policy Analyst", "Community Developer"],
    },
    "Management & Leadership": {
        "emoji": "🧭",
        "desc":  "You naturally organise people, delegate well, and keep teams aligned. Operations, project management, and executive roles are where you'll flourish.",
        "roles": ["Project Manager", "Operations Manager", "Team Lead", "Startup Founder"],
    },
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/survey")
def survey():
    return render_template("survey.html", questions=QUESTIONS, total=len(QUESTIONS))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # Helper: get form value, fallback to "values" list if display label differs
        def get_val(q):
            raw = form.get(q["id"], "")
            if "values" in q:
                # map display option index to actual value
                try:
                    idx = q["options"].index(raw)
                    return q["values"][idx]
                except ValueError:
                    return raw
            return raw

        data = CustomData(
            Bookstore        = get_val(QUESTIONS[0]),
            Curiosity        = get_val(QUESTIONS[1]),
            Flow             = get_val(QUESTIONS[2]),
            Childhood        = get_val(QUESTIONS[3]),
            Pride_Project    = get_val(QUESTIONS[4]),
            Hobbies          = get_val(QUESTIONS[5]),
            Energy           = get_val(QUESTIONS[6]),
            Friend_Help      = get_val(QUESTIONS[7]),
            Math             = int(form.get("Math", 5)),
            Language         = int(form.get("Language", 5)),
            Creativity       = int(form.get("Creativity", 5)),
            Management       = int(form.get("Management", 5)),
            Group_Role       = get_val(QUESTIONS[12]),
            Work_Rhythm      = get_val(QUESTIONS[13]),
            Thinking         = get_val(QUESTIONS[14]),
            Structure        = get_val(QUESTIONS[15]),
            Decision         = get_val(QUESTIONS[16]),
            Job_Choice       = get_val(QUESTIONS[17]),
            No_Money_Problem = get_val(QUESTIONS[18]),
            Fulfillment      = get_val(QUESTIONS[19]),
            Success          = get_val(QUESTIONS[20]),
            Regret           = get_val(QUESTIONS[21]),
            Ideal_Week       = get_val(QUESTIONS[22]),
            Environment      = get_val(QUESTIONS[23]),
            Failure          = get_val(QUESTIONS[24]),
        )

        df       = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        results  = pipeline.predict(df)

        # Enrich results with track info
        enriched = []
        for r in results:
            info = TRACK_INFO.get(r["track"], {})
            enriched.append({**r, **info})

        logging.info(f"Prediction served: {[r['track'] for r in results]}")
        return render_template("result.html", results=enriched)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)