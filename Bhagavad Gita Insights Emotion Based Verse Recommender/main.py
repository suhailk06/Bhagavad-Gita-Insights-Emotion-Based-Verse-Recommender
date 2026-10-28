from pathlib import Path
import re
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Bhagavad Gita Insights")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

TAGS = {
    "angry":        ["angry", "mad", "furious", "rage", "hate", "irritated", "annoyed", "frustrated", 
                     "pissed", "outraged", "bitter", "resentful", "hostile", "aggravated", "fuming",
                     "seething", "infuriated", "livid", "fed up", "exasperated", "displeased",
                     " pissed off", "triggered", "seeing red", "done with", "sick of", "blood boiling"],
    "sad":          ["sad", "cry", "unhappy", "depressed", "down", "miserable", "tears", "hopeless", 
                     "grief", "heartbroken", "gloomy", "sorrow", "downcast", "upset", "blue",
                     "devastated", "glum", "crestfallen", "heavy-hearted", "despair", "melancholy",
                     "crying", "weeping", "low", "hurting", "heavy", "broken", "empty inside"],
    "fear":         ["fear", "scared", "afraid", "terrified", "panic", "worried", "anxious", "nervous", 
                     "dread", "frightened", "jittery", "alarmed", "edgy", "paranoid", "phobia",
                     "petrified", "threatened", "uneasy", "apprehensive", "timid", "dreading",
                     "freaking out", "sweating", "stressed out", "what if", "doom", "scared to death"],
    "confused":     ["confused", "lost", "unclear", "unsure", "puzzled", "dont know", "do not know",
                     "no idea", "what to do", "cant figure", "perplexed", "baffled", "clueless", "mixed up",
                     "disoriented", "befuddled", "muddled", "mystified", "turned around",
                     "makes no sense", "completely lost", "head spinning", "over my head", "at a loss"],
    "stressed":     ["stressed", "pressure", "overwhelmed", "burnout", "tension", "too much",
                     "exam", "deadline", "no time", "running out", "strained", "swamped", "burdened", "frazzled",
                     "cracked under", "stretched thin", "panicking", "hiccup", "chaotic", "pressurized",
                     "drowning", "swimming in it", "cant breathe", "crushing", "piling up", "sanity"],
    "demotivated":  ["demotivated", "lazy", "give up", "no energy", "unmotivated", "tired",
                     "cant focus", "procrastinat", "exhausted", "apathy", "sluggish", "drained", "burned out", "lethargic",
                     "uninspired", "stuck", "fatigued", "listless", "inert", "burnt out", "slump",
                     "cant be bothered", "whats the point", "heavy limbs", "dragging", "zero drive"],
    "guilt":        ["guilt", "guilty", "sorry", "blame", "regret", "my fault", "should have",
                     "ashamed", "remorse", "conscience", "self-blame", "at fault", "penitent",
                     "accountable", "regretting", "punishing myself", "bad conscience",
                     "burden of proof", "haunted", "wish i hadn't", "shouldnt have", "my bad"],
    "attachment":   ["attachment", "attached", "miss", "cling", "cant let go", "dependent", "obsessed",
                     "holding on", "needy", "fixated", "codependent", "craving", "inseparable",
                     "hooked", "reliant", "can't live without", "possessive",
                     "stalking", "cant stop thinking", "addicted to them", "won't leave my mind"],
    "failure":      ["failure", "failed", "loser", "not good enough", "useless", "worthless",
                     "didnt start", "did not start", "didnt study", "did not study", "wasted",
                     "inadequate", "disappointment", "incompetent", "flop", "ruined",
                     "mess up", "messed up", "falling behind", "zero", "good for nothing",
                     "total disaster", "screw up", "screwed up", "imposter", "cant do anything right"],
    "relationship": ["relationship", "partner", "boyfriend", "girlfriend", "friend", "family",
                     "love", "breakup", "crush", "spouse", "marriage", "husband", "wife", "dating", "ex",
                     "significant other", "romance", "couple", "fiance", "colleague", "relative",
                     "situationship", "talking stage", "toxic", "commitment", "cheating", "heartbreak"],
    "greed":        ["greed", "greedy", "want more", "never enough", "selfish", "possessive",
                     "materialistic", "covetous", "hoarding", "ambitious for more", "grasping",
                     "insatiable", "gluttonous", "self-serving", "all mine", "stingy",
                     "taking everything", "power hungry", "gimme", "unforgivingly rich"],
    "purpose":      ["purpose", "meaning", "why am i", "goal", "direction", "lost in life",
                     "pointless", "no point", "aimless", "directionless", "calling", "significance", "emptiness",
                     "vacant", "what's the use", "whats the point", "no vision", "adrift", "stuck in life",
                     "existential", "what am i doing", "walking dead", "void", "hollow"],
    "ego":          ["ego", "pride", "arrogant", "superior", "i am right", "selfish",
                     "hubris", "self-importance", "conceited", "boastful", "stubborn",
                     "narcissistic", "know-it-all", "haughty", "vain", "full of myself", "egoistic",
                     "high horse", "too proud", "cant admit", "my way or the highway"],
    "decision":     ["decision", "decide", "choice", "choose", "dilemma", "what to do",
                     "cant decide", "should i", "crossroads", "options", "hesitating", "torn",
                     "conflicted", "indecisive", "ambivalent", "weighing", "pro and con",
                     "flip a coin", "stuck between", "two minds", "paralysis of choice"],
    "loneliness":   ["lonely", "alone", "isolated", "no one", "empty", "left out", "nobody",
                     "solitude", "abandoned", "detached", "alienated", "friendless", "desolate",
                     "unwanted", "outsider", "by myself", "unseen", "forgotten",
                     "all by myself", "isolated", "nobody cares", "invisible", "no text messages"],
}

SOLUTIONS = {
    "angry": "Pause and take 3 deep breaths before reacting. Step away from the situation, let your body cool down, and then express your feelings calmly using 'I' statements.",
    
    "sad": "Allow yourself to feel it without judgment. Talk to someone you trust, cry if you need to, and do one small comforting thing like taking a warm shower or listening to calm music.",
    
    "fear": "Ground yourself in the present moment by naming 5 things you can see and 4 things you can touch. Ask yourself: 'Is this danger real right now, or is it just a worry?'",
    
    "confused": "Write down everything that is in your head on a piece of paper. Break the big problem into smaller, bite-sized pieces and focus only on the very next step.",
    
    "stressed": "Stop multitasking and pick just one task. Do a quick physical stretch, drink a glass of water, and remember you don't have to fix everything all at once.",
    
    "demotivated": "Lower the bar completely. Tell yourself you only need to work on the task for 2 minutes. Often, just starting is enough to build a tiny bit of momentum.",
    
    "guilt": "Acknowledge what went wrong, apologize or make amends if you can, and then consciously forgive yourself. Learn the lesson and let the heavy baggage go.",
    
    "attachment": "Shift your focus back to yourself. Invest your time into your own hobbies, friends, and personal goals so your happiness comes from within, not just one source.",
    
    "failure": "Separate your self-worth from the outcome. Treat this as data or feedback rather than a permanent label, and ask: 'What can I adjust and try differently next time?'",
    
    "relationship": "Communicate your feelings honestly and openly without playing blame games. Listen actively to the other person, and set healthy boundaries if needed.",
    
    "greed": "Practice gratitude daily by writing down three things you already have. Remind yourself that peace and contentment come from appreciating the present, not collecting more.",
    
    "purpose": "Stop searching for a grand, massive destiny. Instead, focus on doing small, meaningful acts of kindness or curiosity today—purpose is built one small step at a time.",
    
    "ego": "Practice humility by actively listening to someone else's perspective without interrupting. Remind yourself that making mistakes is normal and doesn't lower your worth.",
    
    "decision": "Flip a coin. While it's in the air, you will often suddenly realize which outcome you are secretly hoping for. If not, pick the best option available and trust that you can adapt.",
    
    "loneliness": "Reach out to a friend or family member, even with a simple text message. Alternatively, try doing a public activity you enjoy, like going to a café or a library, to feel connected to the world around you."
}


def normalize(text: str) -> str:
    return text.lower().replace("'", "").replace("’", "")


list_solutions=[]
# def analyze(sentence: str) -> dict[str, int]:
#     sentence = normalize(sentence)
#     scores = {}
#     for tag, keywords in TAGS.items():
#         scores[tag] = int(any(kw in sentence for kw in keywords))
#         if scores[tag] > 0:
#             print(scores)
#     return scores



def analyze(sentence: str) -> dict[str, int]:
    sentence = normalize(sentence)
    scores = {}
    
    for tag, keywords in TAGS.items():
        matched = False
        for kw in keywords:
            if " " in kw:
                # If the keyword is a multi-word phrase, check substring match
                if kw in sentence:
                    matched = True
                    break
            else:
                # If it's a single word, ensure it matches as a whole word to prevent false positives
                if re.search(r'\b' + re.escape(kw) + r'\b', sentence):
                    matched = True
                    break
        scores[tag] = int(matched)
        
    return scores


def tags_to_vector(scores: dict[str, int]) -> list[int]:
    return [scores[tag] for tag in TAGS]


def load_model():
    dataset = pd.read_excel(BASE_DIR / "dataset.xlsx")
    X_df = dataset.iloc[:, 9:].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X_df.values
    Y = dataset.iloc[:, 4:7].astype(str).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    classifier = KNeighborsClassifier(n_neighbors=1, metric="minkowski", p=2)
    classifier.fit(X_scaled, Y)
    return scaler, classifier, X.shape[1]


SCALER, CLASSIFIER, N_FEATURES = load_model()


def solve(text: str):
    scores = analyze(text)
    vector = tags_to_vector(scores)

    if len(vector) != N_FEATURES:
        raise ValueError(
            f"TAGS produce {len(vector)} features, but dataset.xlsx has {N_FEATURES} feature columns. "
            "Make sure TAGS order/count matches dataset.iloc[:, 9:14]."
        )

    prediction = CLASSIFIER.predict(SCALER.transform([vector]))[0]

    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()

    # Extract human-readable actionable solutions based on detected emotions
    matched_solutions = {
        tag: SOLUTIONS[tag]
        for tag, score in scores.items()
        if score > 0 and tag in SOLUTIONS
    }

    return scores, prediction, matched_solutions


@app.post("/")
async def submit_content(request: Request, content: str = Form("")):
    scores, prediction, solutions = solve(content)
    return templates.TemplateResponse(
        request,
        "content.html",
        {
            "content": content,
            "scores": scores,
            "prediction": prediction,
            "solutions": solutions,
        },
    )


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


