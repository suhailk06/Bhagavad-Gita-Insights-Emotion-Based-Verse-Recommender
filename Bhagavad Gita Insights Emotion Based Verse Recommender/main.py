from pathlib import Path

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


def normalize(text: str) -> str:
    return text.lower().replace("'", "").replace("’", "")


def analyze(sentence: str) -> dict[str, int]:
    sentence = normalize(sentence)
    scores = {}
    for tag, keywords in TAGS.items():
        scores[tag] = int(any(kw in sentence for kw in keywords))
    return scores


def tags_to_vector(scores: dict[str, int]) -> list[int]:
    # Important: this order must match the dataset feature-column order.
    return [scores[tag] for tag in TAGS]


def load_model():
    dataset = pd.read_excel(BASE_DIR / "dataset.xlsx")
    X_df = dataset.iloc[:, 9:].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X_df.values
    Y = dataset.iloc[:, 4:9].astype(str).values
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

    return scores, prediction


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


@app.post("/")
async def submit_content(request: Request, content: str = Form("")):
    scores, prediction = solve(content)
    return templates.TemplateResponse(
        request,
        "content.html",
        {
            "content": content,
            "scores": scores,
            "prediction": prediction,
        },
    )