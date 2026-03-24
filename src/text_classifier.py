"""
text_classifier.py
──────────────────
Keyword-based classifier that maps free-text survey answers to the
fixed category values used during model training.

WHY THIS EXISTS
---------------
Questions 5, 6, 7, 15, 16, 18, 19, 21, 23 are open-ended in the original
survey — users type their own answers. The ML model was trained on 10
fixed string values per column (pre-bucketed in the dataset).

This module bridges the gap: it takes any free-text answer and maps it
to the closest matching training value using keyword scoring.

HOW IT WORKS
------------
Each column has a list of (label, keywords) pairs. The classifier counts
how many keywords from each label appear in the user's text. The label
with the highest count wins. If nothing matches, the first (most common)
label is used as a safe fallback.

USAGE
-----
    from src.text_classifier import classify_free_text

    result = classify_free_text(
        text="I built a patient tracking app for a hospital",
        column="Pride_Project"
    )
    # Returns: "Completed internship project"
"""

from typing import Optional

# ── Keyword maps ──────────────────────────────────────────────────────────────
# Format: list of (training_label, [keyword_fragments])
# keyword_fragments use partial matching — "teach" matches "teaching", "teacher"
# Order within each column matters only for tie-breaking (first = fallback)

KEYWORD_MAPS: dict[str, list[tuple[str, list[str]]]] = {

    # Q5 — Pride Project
    "Pride_Project": [
        ("Organised a social drive",           ["social drive","blood donation","charity","volunteer","donation","awareness campaign","ngo","community service","social cause"]),
        ("Published a blog series",            ["blog","article","post","newsletter","medium","published","wrote","content","writing","editorial"]),
        ("Built an automation script",         ["automat","script","bot","scraper","pipeline","workflow","cli","program","tool","system"]),
        ("Built a personal website",           ["website","web app","portfolio","frontend","html","css","personal site","landing page"]),
        ("Led a college event",                ["event","fest","college","university","organised","managed","led","coordinated","team lead","hackathon organis"]),
        ("Completed a research mini-project",  ["research","study","survey","analysis","paper","thesis","report","findings","dataset","experiment"]),
        ("Won a hackathon",                    ["hackathon","competition","contest","won","prize","award","coding challenge","first place","national"]),
        ("Completed internship project",       ["intern","placement","job","work experience","company","office","corporate","hospital","clinic","medical","health","finance","bank"]),
        ("Created a mobile app",               ["mobile app","android","ios","flutter","react native","application","app store","play store"]),
        ("Designed a prototype model",         ["prototype","model","mockup","figma","3d","hardware","circuit","product design","industrial"]),
    ],

    # Q6 — Hobbies
    "Hobbies": [
        ("Sports, Fitness, Yoga",              ["sport","fitness","gym","yoga","run","football","cricket","basketball","health","workout","swim","badminton","cycling","athlet"]),
        ("Gaming, Streaming, Editing",         ["gaming","game","stream","edit","video","youtube","twitch","content creat","vlog","podcast creat","reel"]),
        ("Finance tracking, Investing",        ["finance","invest","stock","crypto","budget","money","trading","portfolio","sip","mutual fund","market","shares"]),
        ("Art, Sketching, Design",             ["art","sketch","draw","paint","design","illustrat","graphic","aesthetic","ui","ux","visual"]),
        ("DIY Projects, Robotics",             ["diy","robot","build","make","electronics","circuit","arduino","raspberry","hardware","tinker","3d print"]),
        ("Writing, Blogging, Music",           ["writ","blog","music","sing","guitar","poem","story","lyric","instrument","compose","novel","fiction"]),
        ("Travel, Photography, Journaling",    ["travel","photo","journal","explor","backpack","nature","hike","camp","wander","landscape","trek"]),
        ("Podcast listening, Note-taking",     ["podcast","note","listen","audiobook","journal","reflect","self-improv","productivity","mindful"]),
        ("Reading, Coding, Chess",             ["cod","program","chess","read","algorithm","competitive","leetcode","open source","developer","hackathon"]),
        ("Volunteering, Teaching",             ["volunteer","teach","tutor","mentor","ngo","social work","community","educate","coach","guide"]),
    ],

    # Q7 — Energy Barometer
    "Energy": [
        ("Solving complex problems energizes me",      ["solv","problem","complex","challenge","analys","debug","logic","puzzle","algorithm","difficult","technical"]),
        ("Creative discussions excite me",             ["creative","idea","innovat","design","artistic","express","imagine","startup","brainstorm with","concept"]),
        ("Learning new concepts motivates me",         ["learn","concept","knowledge","study","understand","skill","grow","new things","course","read"]),
        ("Building things keeps me engaged",           ["build","make","construct","develop","engineer","hardware","product","create something","prototype"]),
        ("Helping people gives me energy",             ["help","people","support","care","serve","assist","community","impact","others","human"]),
        ("Collaborative brainstorming energizes me",   ["collaborat","team","together","discuss","group","with friend","peer","colleague","joint"]),
        ("Exploring new ideas excites me",             ["explor","curious","discover","new","research","unknown","what if","question","wonder"]),
        ("Deep focused work energizes me",             ["focus","deep","alone","concentrate","flow","quiet","solo","immerse","undisturb","sink into"]),
        ("Teaching others feels energizing",           ["teach","mentor","explain","train","educate","guide","junior","tutor","sharing knowledge","help others learn","lecture"]),
        ("Strategic planning excites me",              ["strateg","plan","organis","manage","lead","goal","execut","vision","roadmap","priorit"]),
    ],

    # Q15 — Job Choice (non-negotiable test)
    "Job_Choice": [
        ("Preferred work-life balance over pay",       ["balance","life","wellbeing","health","time","personal","boundary","family time","rest","quality of life"]),
        ("Chose higher salary for security",           ["salary","pay","money","compens","financial security","high paying","package","ctc","income"]),
        ("Opted for flexibility and autonomy",         ["flexib","autonom","freedom","remote","independent","own time","wfh","hybrid","no micromanage"]),
        ("Selected mission-driven organization",       ["mission","purpose","impact","cause","ngo","social","values","meaningful","driven","believe in"]),
        ("Chose learning and growth over salary",      ["learn","grow","skill","development","opport","mentor","training","career growth","upskill"]),
        ("Preferred remote-friendly role",             ["remote","work from home","wfh","online","virtual","anywhere","no commute","location"]),
        ("Chose stability due to family needs",        ["stab","family","secure","safe","predictable","consistent","dependable","steady","support family"]),
        ("Preferred challenging work environment",     ["challeng","difficult","complex","push","stretch","demanding","hard","stimulating","competitive"]),
        ("Selected role aligned with passion",         ["passion","love","interest","fulfil","meaningful","enjoy","excited","care about"]),
        ("Selected role with long-term impact",        ["long.term","impact","future","legacy","lasting","change","generational","10 years","big picture"]),
    ],

    # Q16 — No-Money Problem
    "No_Money_Problem": [
        ("Solve healthcare affordability",             ["health","medical","hospital","doctor","medicine","afford","cure","disease","patient","clinic","treatment","cancer","drug"]),
        ("Improve education access",                   ["educat","school","learn","literacy","student","teach","access","college","university","children learn"]),
        ("Improve mental health awareness",            ["mental health","anxiety","depress","wellbeing","therapy","stress","awareness","suicide","burnout","trauma"]),
        ("Reduce unemployment",                        ["unemploy","job","work","career","employ","livelihood","income","jobless","skill gap"]),
        ("Build sustainable technology",               ["technolog","sustain","clean energy","renewable","green tech","climate tech","ev","solar","emission"]),
        ("Improve environmental sustainability",       ["environment","climate","pollution","carbon","waste","nature","planet","ecosystem","deforest","ocean"]),
        ("Improve public safety systems",              ["safety","crime","police","disaster","emergency","protect","security","violenc","accident"]),
        ("Improve financial literacy",                 ["financ","money","literacy","invest","budget","economic","poverty","wealth","savings","debt"]),
        ("Reduce digital divide",                      ["digital","internet","access","tech gap","connectivity","rural","inclusion","device","bandwidth"]),
        ("Support small businesses",                   ["small business","entrepreneur","startup","local","msme","shop","owner","vendor","artisan","self-employ"]),
    ],

    # Q18 — Defining Success
    "Success": [
        ("Doing meaningful work",                      ["meaning","purpose","matter","contribut","significant","worthwhile","fulfil","reason","difference"]),
        ("Personal growth and learning",               ["grow","learn","develop","skill","knowledge","improve","evolv","better","version","potential"]),
        ("Positive impact on society",                 ["impact","society","world","change","community","contribut","generation","humanity","people"]),
        ("Helping others succeed",                     ["help","others","impact","people","uplift","serve","teach","mentor","enable","empower"]),
        ("Financial independence",                     ["financ","money","wealth","independ","freedom","rich","afford","passive income","retire"]),
        ("Creating valuable solutions",                ["creat","build","solut","product","innovat","value","design","launch","make"]),
        ("Recognition for expertise",                  ["recognit","expert","known","respect","authority","leader","reputation","award","renowned"]),
        ("Balanced and fulfilling life",               ["balanc","fulfil","happy","content","wholesome","peaceful","harmony","complete"]),
        ("Long-term satisfaction",                     ["satisf","content","peace","long.term","stable","comfortable","no regret","look back"]),
        ("Freedom of choice",                          ["freedom","choice","autonom","own","decide","control","independ","terms","lifestyle"]),
    ],

    # Q19 — Career Feel (Party Test)
    "Career_Feel": [
        ("Impact",       ["impact","change","matter","difference","meaningful","contribut","transforming","legacy"]),
        ("Creativity",   ["creat","innovat","art","design","imagin","original","express","built","make"]),
        ("Leadership",   ["leader","inspir","guide","vision","lead","influence","mentor","drove","pioneer"]),
        ("Innovation",   ["innovat","new","disrupt","pioneer","invent","forward","technology","revolution","first"]),
        ("Trust",        ["trust","honest","transparent","authentic","open","sincere","reliable","dependable"]),
        ("Excellence",   ["excellen","quality","best","mastery","perfection","high standard","outstanding","top"]),
        ("Integrity",    ["integrit","honest","ethic","principl","value","authentic","moral","fair"]),
        ("Compassion",   ["compassion","empathy","care","kind","support","help","love","gentle","nurtur","warm"]),
        ("Reliability",  ["reliab","depend","consist","stable","promise","deliver","on time","follow through"]),
        ("Respect",      ["respect","dignit","honor","appreciat","recognit","admire","look up"]),
    ],

    # Q21 — Ideal Week Blueprint
    "Ideal_Week": [
        ("Deep work with short meetings",              ["deep","focus","meeting","short","concentrated","minimal","interrupt","code","build","write"]),
        ("Structured weekdays, flexible weekends",     ["structur","routine","weekday","weekend","plan","organis","9 to 5","schedule"]),
        ("Collaborative start, solo execution",        ["collaborat","team","solo","execut","together then alone","standup","morning meeting"]),
        ("Creative work with reviews",                 ["creat","design","review","feedback","iterati","art","write","refine","critique"]),
        ("Field work mixed with planning",             ["field","outdoor","plan","visit","site","travel","mixed","on ground","client"]),
        ("Mostly independent focused work",            ["independ","solo","alone","self","quiet","autonomous","no meeting","by myself"]),
        ("Project-driven intense weeks",               ["project","intense","sprint","deadline","deliver","driven","goal","crunch","milestone"]),
        ("Balanced mix of work and learning",          ["balanc","mix","learn","work","both","blend","variety","half","also"]),
        ("Flexible schedule with focus blocks",        ["flexib","schedule","block","focus","own time","adapt","when i want","pomodoro"]),
        ("Learning-heavy exploratory week",            ["learn","explor","research","course","curious","discover","study","read","experiment"]),
    ],

    # Q23 — Defining Failure
    "Failure": [
        ("Loss of motivation",         ["motivat","passion","drive","meaning","interest","purpose","quit","give up","lose will"]),
        ("Lack of impact",             ["impact","matter","meaningless","purpose","nothing","useless","irrelevant","no difference","wasted"]),
        ("Ethical compromise",         ["ethic","compromise","value","moral","integrity","wrong","corrupt","against principle","dishonest"]),
        ("Ignoring feedback",          ["feedback","ignore","listen","stubborn","ego","advice","input","not hear","close-minded"]),
        ("Stagnation without learning",["stagnate","stuck","grow","learn","progress","develop","static","same place","no improvement"]),
        ("Not adapting to change",     ["adapt","change","rigid","stuck","resist","flexible","evolv","outdated","refuse to change"]),
        ("Poor time management",       ["time","deadline","procrastinat","late","schedule","priorit","manage","miss deadline","delay"]),
        ("Poor execution",             ["execut","deliver","finish","complete","follow through","action","result","never finish","start but"]),
        ("Fear of taking risks",       ["risk","fear","safe","comfort zone","afraid","avoid","hesit","never try","play it safe"]),
        ("Burnout",                    ["burnout","exhaust","overwhelm","too much","drain","overwork","tired","health","collapse"]),
    ],
}

# Columns that use the classifier (free-text questions)
FREE_TEXT_COLUMNS = list(KEYWORD_MAPS.keys())


def classify_free_text(text: str, column: str) -> str:
    """
    Maps a free-text answer to the closest training label for a given column.

    Parameters
    ----------
    text   : str  — the user's raw free-text input
    column : str  — the column name (e.g. 'Pride_Project', 'Energy')

    Returns
    -------
    str — one of the 10 fixed training labels for that column,
          or the original text if column is not in the classifier
    """
    if column not in KEYWORD_MAPS:
        return text  # structured column — return as-is

    text_lower = text.lower().strip()
    mapping = KEYWORD_MAPS[column]

    best_label = mapping[0][0]  # safe fallback = most common class
    best_score = 0

    for label, keywords in mapping:
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def classify_row(row: dict) -> dict:
    """
    Applies classify_free_text to all free-text columns in a row dict.
    Structured columns (radio / likert) are returned unchanged.

    Parameters
    ----------
    row : dict — survey response with column names as keys

    Returns
    -------
    dict — same row with free-text columns replaced by training labels
    """
    result = dict(row)
    for col in FREE_TEXT_COLUMNS:
        if col in result and isinstance(result[col], str):
            result[col] = classify_free_text(result[col], col)
    return result