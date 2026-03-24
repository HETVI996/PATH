import sys
from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
from src.logger import logging
app = Flask(__name__)

# Database is optional — only active if database.py exists
try:
    from database import init_db, save_response
    init_db()
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    def save_response(*args, **kwargs): pass

# ── Survey questions ──────────────────────────────────────────────────────────
QUESTIONS = [
    {
        "id":"Bookstore","number":1,"title":"The bookstore test",
        "question":"If you walked into a bookstore with unlimited time and money, where do you go first?",
        "type":"radio","options":["Fiction","Science/Technology","History","Self-Help"],
    },
    {
        "id":"Curiosity","number":2,"title":"Your curiosity",
        "question":"Outside work or school, what topics do you find yourself reading or watching videos about just for fun?",
        "type":"radio",
        "options":["Technology","Health","Travel","Finance","Science",
                   "Arts / Culture","Sports / Fitness","Justice / Law","Politics / Policy"],
    },
    {
        "id":"Flow","number":3,"title":"Flow state",
        "question":"Which activity makes you completely lose track of time?",
        "type":"radio",
        "options":["Writing","Coding","Sports","Games","Experiments",
                   "Drawing / Making","Debating","Reading / Writing"],
    },
    {
        "id":"Childhood","number":4,"title":"Earliest interests",
        "question":"Looking back at childhood, what did you love doing most?",
        "type":"radio","options":["Building","Storytelling","Organizing","Exploring"],
    },
    {
        "id":"Pride_Project","number":5,"title":"Pride project",
        "question":"Think about a project or accomplishment you are genuinely proud of. Describe it briefly.",
        "type":"freetext","hint":"e.g. Built a hospital app, Won a law moot court, Directed a theatre production…",
        "placeholder":"Describe your proudest achievement…",
    },
    {
        "id":"Hobbies","number":6,"title":"Your hobbies",
        "question":"What are your main hobbies or interests outside academics?",
        "type":"radio",
        "options":["Art, Sketching, Design","DIY Projects, Robotics","Finance tracking, Investing",
                   "Gaming, Streaming, Editing","Podcast listening, Note-taking","Reading, Coding, Chess",
                   "Sports, Fitness, Yoga","Travel, Photography, Journaling","Volunteering, Teaching",
                   "Writing, Blogging, Music","Science experiments, Nature observation",
                   "Painting, Sculpting, Music","Debating / MUNs","Current affairs, Reading"],
    },
    {
        "id":"Energy","number":7,"title":"Energy barometer",
        "question":"What conversations, tasks, or subjects leave you feeling energised or excited?",
        "type":"freetext","hint":"e.g. Solving hard problems, Teaching others, Debating policy issues…",
        "placeholder":"Describe what energises you…",
    },
    {
        "id":"Friend_Help","number":8,"title":"Friend-sourced skill",
        "question":"What do your friends or family consistently ask you for help with?",
        "type":"radio","options":["Technology","Organizing","Emotional Support","Problem-Solving"],
    },
    {
        "id":"Math","number":9,"title":"Natural ease — mathematics",
        "question":"How intuitively easy did mathematics feel for you? (1 = very hard, 10 = effortless)",
        "type":"likert","min":1,"max":10,
    },
    {
        "id":"Language","number":10,"title":"Natural ease — language",
        "question":"How intuitively easy did language / writing feel for you? (1 = very hard, 10 = effortless)",
        "type":"likert","min":1,"max":10,
    },
    {
        "id":"Creativity","number":11,"title":"Natural ease — creativity",
        "question":"How naturally creative do you feel? (1 = not at all, 10 = extremely)",
        "type":"likert","min":1,"max":10,
    },
    {
        "id":"Management","number":12,"title":"Natural ease — management",
        "question":"How naturally do you take charge and organise people and tasks? (1 = never, 10 = always)",
        "type":"likert","min":1,"max":10,
    },
    {
        "id":"Group_Role","number":13,"title":"Problem-solving role",
        "question":"When working in a group on a difficult problem, what role do you naturally play?",
        "type":"radio",
        "options":["Idea generator — brainstorming creative, out-of-box ideas",
                   "Orchestrator — organising the plan and delegating tasks",
                   "Detective — researching information and finding the facts",
                   "Builder — executing the plan and creating the final solution"],
        "values":["Idea generator","Orchestrator","Detective","Builder"],
    },
    {
        "id":"Work_Rhythm","number":14,"title":"Work rhythm",
        "question":"How do you prefer to work?",
        "type":"radio",
        "options":["Deeply focus on one big project for a long period",
                   "Enjoy the variety of juggling many smaller tasks at once"],
        "values":["One big project","Many small tasks"],
    },
    {
        "id":"Thinking","number":15,"title":"Thinking style",
        "question":"When do you do your best thinking?",
        "type":"radio",
        "options":["Brainstorming and collaborating live with the team",
                   "Working alone in a quiet space to think deeply",
                   "A mix — collaborating for ideas, then working solo to execute"],
        "values":["Team","Alone","Mix"],
    },
    {
        "id":"Structure","number":16,"title":"Structure vs freedom",
        "question":"Which work environment sounds more appealing?",
        "type":"radio",
        "options":["A job with clear, predictable routines and well-defined tasks",
                   "One with a flexible schedule and freedom to experiment"],
        "values":["Structure","Freedom"],
    },
    {
        "id":"Decision","number":17,"title":"Decision-making style",
        "question":"When making an important decision, do you primarily rely on…",
        "type":"radio",
        "options":["Head — objective data, logic, and pros/cons",
                   "Heart — intuition, personal values, and impact on people"],
        "values":["Head","Heart"],
    },
    {
        "id":"Job_Choice","number":18,"title":"The non-negotiable test",
        "question":"Imagine two job offers — Job A has a higher salary, Job B has something you deeply value. Which would you choose and why?",
        "type":"freetext","hint":"e.g. I chose B for the learning opportunity over a higher salary.",
        "placeholder":"Describe your choice and reasoning…",
    },
    {
        "id":"No_Money_Problem","number":19,"title":"The no-money test",
        "question":"If money were no object, what problem in the world would you dedicate your life to solving?",
        "type":"freetext","hint":"e.g. I would cure a neglected disease, reform the justice system, preserve traditional arts…",
        "placeholder":"Describe the problem you'd solve…",
    },
    {
        "id":"Fulfillment","number":20,"title":"Contribution vs creation",
        "question":"Which feels more fulfilling to you?",
        "type":"radio",
        "options":["Directly help, teach or serve people",
                   "Create something new that people can use or experience"],
        "values":["Help people","Create"],
    },
    {
        "id":"Success","number":21,"title":"Defining success",
        "question":"At the end of your life, what needs to be true for it to feel like a success?",
        "type":"freetext","hint":"e.g. Having helped thousands of people, creating a body of work I'm proud of…",
        "placeholder":"Describe what success means to you…",
    },
    {
        "id":"Career_Feel","number":22,"title":"The party test",
        "question":"What one feeling do you want people to associate with your career story in 20 years?",
        "type":"freetext","hint":"e.g. Impact, Compassion, Innovation, Excellence…",
        "placeholder":"One word or feeling…",
    },
    {
        "id":"Regret","number":23,"title":"Regret minimisation",
        "question":"Which long-term regret would you rather avoid?",
        "type":"radio",
        "options":["The regret of not taking a major risk for higher reward",
                   "The regret of not choosing a path that guaranteed stability"],
        "values":["Risk","Stability"],
    },
    {
        "id":"Ideal_Week","number":24,"title":"Ideal work week",
        "question":"Design your ideal work week. What does it look like day to day?",
        "type":"freetext","hint":"e.g. Patient care in the mornings, research in the afternoons, supervision sessions weekly…",
        "placeholder":"Describe your ideal work week…",
    },
    {
        "id":"Environment","number":25,"title":"Work environment",
        "question":"Select your optimal work environment:",
        "type":"radio",
        "options":["Corporate","Field-based","Startup","Solo",
                   "Research lab","Studio / Gallery","Outdoor / Training ground",
                   "Courtroom / Chambers","Government office"],
    },
    {
        "id":"Failure","number":26,"title":"Defining failure",
        "question":"What would a significant failure look like in your career, and how would you recover?",
        "type":"freetext","hint":"e.g. Burning out without impact. I'd recover by seeking mentorship and switching direction.",
        "placeholder":"Describe failure and recovery…",
    },
]

# ── Career track info for results page ───────────────────────────────────────
TRACK_INFO = {
    "Technology & Engineering": {
        "emoji":"⚙️",
        "desc":"You thrive building systems, writing code, and solving hard technical problems. Software engineering, product development, and deep tech are your domain.",
        "roles":["Software Engineer","Product Developer","DevOps Engineer","Systems Architect"],
    },
    "Data & Research": {
        "emoji":"🔍",
        "desc":"You have a detective's curiosity — digging into data, finding patterns, and turning insights into decisions. Analytics, data science, and research roles suit you.",
        "roles":["Data Scientist","Research Analyst","ML Engineer","Business Intelligence Analyst"],
    },
    "Natural Sciences": {
        "emoji":"🔬",
        "desc":"You are driven by understanding the natural world. Biology, chemistry, physics, and environmental science energise you. Labs and fieldwork are your home.",
        "roles":["Research Scientist","Biotechnologist","Environmental Scientist","Lab Analyst"],
    },
    "Creative & Design": {
        "emoji":"✏️",
        "desc":"You communicate through stories, visuals, and ideas. Writing, content, UX, and brand design let you turn imagination into things people love.",
        "roles":["UX/UI Designer","Content Strategist","Brand Designer","Creative Director"],
    },
    "Fine Arts & Performing Arts": {
        "emoji":"🎭",
        "desc":"You express yourself through art, performance, music, or film. Gallery work, theatre, music production, and creative residencies are where you belong.",
        "roles":["Visual Artist","Performing Artist","Musician","Film Director","Art Curator"],
    },
    "Interior Design & Architecture": {
        "emoji":"🏛️",
        "desc":"You shape physical spaces — turning layouts, materials, and light into experiences. Interior design, architecture, and spatial planning are your craft.",
        "roles":["Interior Designer","Architect","Space Planner","Set Designer","Design Consultant"],
    },
    "Business & Finance": {
        "emoji":"📈",
        "desc":"You think in numbers, strategies, and outcomes. Finance, consulting, and entrepreneurship let you build and scale things that matter economically.",
        "roles":["Financial Analyst","Business Consultant","Entrepreneur","Investment Analyst"],
    },
    "Healthcare & Wellness": {
        "emoji":"🩺",
        "desc":"Your drive to help people heal makes you a natural in medicine, nursing, public health, or wellness. Clinical practice and community health are your domain.",
        "roles":["Doctor","Nurse","Public Health Specialist","Physiotherapist","Healthcare Manager"],
    },
    "Psychology & Counselling": {
        "emoji":"🧠",
        "desc":"You are drawn to understanding the human mind and helping people grow. Counselling, therapy, research, and mental health advocacy are your calling.",
        "roles":["Psychologist","Counsellor","Therapist","Mental Health Researcher","Social Worker"],
    },
    "Social Impact & Education": {
        "emoji":"🌍",
        "desc":"You want to change systems, educate, and lift communities. NGOs, teaching, policy, and social enterprise let your values and skills converge.",
        "roles":["Educator","NGO Program Manager","Community Developer","Policy Analyst","Social Entrepreneur"],
    },
    "Policy, Law & Public Service": {
        "emoji":"⚖️",
        "desc":"You are drawn to justice, governance, and public systems. Law, policy, government service, and advocacy are where your voice carries the most weight.",
        "roles":["Lawyer","Policy Analyst","Civil Servant","Diplomat","Public Interest Advocate"],
    },
    "Sports & Fitness": {
        "emoji":"🏅",
        "desc":"You are built for performance, discipline, and pushing physical limits. Professional sport, coaching, sports science, and fitness training are your arena.",
        "roles":["Professional Athlete","Sports Coach","Fitness Trainer","Sports Scientist","Athletic Director"],
    },
    "Management & Leadership": {
        "emoji":"🧭",
        "desc":"You naturally organise people, delegate well, and keep teams aligned. Operations, project management, and leadership roles are where you flourish.",
        "roles":["Project Manager","Operations Manager","Team Lead","Startup Founder","COO"],
    },
}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    track_names = list(TRACK_INFO.keys())
    return render_template("index.html", tracks=track_names)


@app.route("/survey")
def survey():
    return render_template("survey.html", questions=QUESTIONS, total=len(QUESTIONS))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        def get_val(q):
            raw = form.get(q["id"], "")
            if "values" in q:
                try:
                    return q["values"][q["options"].index(raw)]
                except (ValueError, IndexError):
                    return raw
            return raw

        data = CustomData(
            Bookstore        = get_val(QUESTIONS[0]),
            Curiosity        = get_val(QUESTIONS[1]),
            Flow             = get_val(QUESTIONS[2]),
            Childhood        = get_val(QUESTIONS[3]),
            Pride_Project    = form.get("Pride_Project", ""),
            Hobbies          = get_val(QUESTIONS[5]),
            Energy           = form.get("Energy", ""),
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
            Job_Choice       = form.get("Job_Choice", ""),
            No_Money_Problem = form.get("No_Money_Problem", ""),
            Fulfillment      = get_val(QUESTIONS[19]),
            Success          = form.get("Success", ""),
            Career_Feel      = form.get("Career_Feel", ""),
            Regret           = get_val(QUESTIONS[22]),
            Ideal_Week       = form.get("Ideal_Week", ""),
            Environment      = get_val(QUESTIONS[24]),
            Failure          = form.get("Failure", ""),
        )

        df       = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        results  = pipeline.predict(df)

        # Save to database
        raw_inputs = {k: v[0] if isinstance(v, list) else v for k, v in form.items()}
        try:
            save_response(raw_inputs=raw_inputs, classified_inputs=raw_inputs, predictions=results)
        except Exception as db_err:
            logging.warning(f"DB save failed (non-fatal): {db_err}")

        enriched = [{**r, **TRACK_INFO.get(r["track"], {})} for r in results]
        return render_template("result.html", results=enriched)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)