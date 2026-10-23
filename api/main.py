from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Bhagavad Gita Insights")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


TAGS = {
    "angry": ["angry", "mad", "furious", "rage", "hate", "irritated", "annoyed", "frustrated"],
    "sad": ["sad", "cry", "unhappy", "depressed", "down", "miserable", "tears", "hopeless"],
    "fear": ["fear", "scared", "afraid", "terrified", "panic", "worried", "anxious", "nervous"],
    "confused": ["confused", "lost", "unclear", "unsure", "puzzled", "dont know", "do not know",
                 "no idea", "what to do", "cant figure"],
    "stressed": ["stressed", "pressure", "overwhelmed", "burnout", "tension", "too much",
                 "exam", "deadline", "no time", "running out"],
    "demotivated": ["demotivated", "lazy", "give up", "no energy", "unmotivated", "tired",
                    "cant focus", "procrastinat"],
    "guilt": ["guilt", "guilty", "sorry", "blame", "regret", "my fault", "should have"],
    "attachment": ["attachment", "attached", "miss", "cling", "cant let go", "dependent", "obsessed"],
    "failure": ["failure", "failed", "loser", "not good enough", "useless", "worthless",
                "didnt start", "did not start", "didnt study", "did not study", "wasted"],
    "relationship": ["relationship", "partner", "boyfriend", "girlfriend", "friend", "family",
                     "love", "breakup", "crush"],
    "greed": ["greed", "greedy", "want more", "never enough", "selfish", "possessive"],
    "purpose": ["purpose", "meaning", "why am i", "goal", "direction", "lost in life",
                "pointless", "no point"],
    "ego": ["ego", "pride", "arrogant", "superior", "i am right", "selfish"],
    "decision": ["decision", "decide", "choice", "choose", "dilemma", "what to do",
                 "cant decide", "should i"],
    "loneliness": ["lonely", "alone", "isolated", "no one", "empty", "left out", "nobody"],
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
    dataset = pd.read_excel(BASE_DIR / "data.xlsx")
    X_df = dataset.iloc[:, 8:].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X_df.values
    Y = dataset.iloc[:, 4:8].astype(str).values
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
            f"TAGS produce {len(vector)} features, but data.xlsx has {N_FEATURES} feature columns. "
            "Make sure TAGS order/count matches dataset.iloc[:, 8:]."
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
    return RedirectResponse(
        request,
        "content.html",
        {
            "content": content,
            "scores": scores,
            "prediction": prediction,
        },
    )