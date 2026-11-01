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

# Severity thresholds for relative intensity (tag_score / total_score)
LOW_THRESHOLD = 0.34      # < 0.34  -> low
HIGH_THRESHOLD = 0.67     # >= 0.67 -> high ; else medium

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

    "angry": {
        "low": (
            "It is understandable to feel angry when something does not "
            "go as expected. Give yourself a little time before reacting. "
            "Identify what upset you and think about how you can express "
            "your feelings without hurting anyone."
        ),
        "medium": (
            "It seems that this situation has affected you deeply. "
            "Instead of allowing anger to control your actions, take "
            "some distance from the situation. Once you feel calmer, "
            "explain what hurt or frustrated you. Focus on resolving "
            "the issue rather than winning the argument."
        ),
        "high": (
            "Your anger sounds overwhelming right now. Stop the argument "
            "and move away from anything that could lead to harm. Avoid "
            "making decisions or confronting anyone while you feel out "
            "of control. Contact someone you trust for support. If you "
            "might hurt yourself or another person, seek immediate help."
        )
    },

    "sad": {
        "low": (
            "You do not have to feel happy all the time. Give yourself "
            "permission to feel sad without judging yourself. Talk to "
            "someone you feel comfortable with or spend some time doing "
            "something that brings you comfort."
        ),
        "medium": (
            "It sounds like something has been weighing on your heart. "
            "Try not to carry everything alone. Share your feelings with "
            "someone you trust. Maintain basic routines such as eating, "
            "sleeping and getting some fresh air. Take things one day "
            "at a time."
        ),
        "high": (
            "I'm sorry things feel so difficult right now. You deserve "
            "support, and you do not need to handle this alone. Contact "
            "someone you trust and consider speaking with a mental "
            "health professional. If you are thinking about harming "
            "yourself, seek immediate emergency support."
        )
    },

    "fear": {
        "low": (
            "Fear is a natural response when something feels uncertain. "
            "Identify exactly what you are afraid of. Think about what "
            "you can prepare for and take one small step instead of "
            "avoiding the situation completely."
        ),
        "medium": (
            "When fear keeps stopping you, divide the situation into "
            "smaller and manageable steps. Prepare for what is within "
            "your control and speak with someone who has experience "
            "with the situation. You do not need to solve everything "
            "at once."
        ),
        "high": (
            "It sounds like fear is becoming difficult to manage. "
            "First, make sure you are in a safe place. Slow down your "
            "breathing and contact someone you trust. Focus on what "
            "you need right now rather than imagining every possible "
            "outcome. If this fear keeps interfering with daily life, "
            "consider professional support."
        )
    },

    "confused": {
        "low": (
            "It is normal not to have every answer immediately. Take "
            "a moment to identify exactly what is confusing you. "
            "Write down your questions and find the information you "
            "need before deciding what to do."
        ),
        "medium": (
            "When several things are happening together, it can be "
            "difficult to think clearly. Separate the problem into "
            "smaller parts. List your options, understand their "
            "possible consequences and discuss the situation with "
            "someone whose judgment you trust."
        ),
        "high": (
            "I understand that you may feel completely stuck. Avoid "
            "rushing into an important decision just to end the "
            "uncertainty. Take a break if possible, gather reliable "
            "information and ask a trusted person or experienced "
            "professional to help you understand your options. "
            "Handle one issue at a time."
        )
    },

    "stressed": {
        "low": (
            "It sounds like you have several things on your mind. "
            "Choose one task that needs attention and start with "
            "that. Take short breaks and give yourself enough time "
            "to rest instead of expecting yourself to do everything "
            "perfectly."
        ),
        "medium": (
            "Trying to manage too many responsibilities can become "
            "exhausting. Write down everything that is troubling you. "
            "Separate urgent tasks from things that can wait. Ask "
            "others to share responsibilities when possible. Make "
            "time for proper sleep, meals and relaxation."
        ),
        "high": (
            "It sounds like the pressure has become too much to "
            "handle alone. Reduce non-essential responsibilities "
            "where possible and tell someone you trust that you "
            "are struggling. Prioritize rest and seek professional "
            "support if stress is affecting your health or daily "
            "functioning."
        )
    },

    "demotivated": {
        "low": (
            "You do not need to feel motivated every day. Choose "
            "one small thing you can complete today. Starting with "
            "a manageable task can help you regain a sense of "
            "progress without putting too much pressure on yourself."
        ),
        "medium": (
            "It can be discouraging when your effort does not produce "
            "the results you expected. Think about whether your goal "
            "is still meaningful to you or whether your approach "
            "needs to change. Set a realistic target and recognize "
            "each step you complete."
        ),
        "high": (
            "It sounds like you have been feeling exhausted or "
            "disconnected from the things that once mattered to "
            "you. Do not blame yourself for this. Reach out to "
            "someone you trust and consider professional support, "
            "especially if this loss of interest or energy continues "
            "or affects your daily life."
        )
    },

    "guilt": {
        "low": (
            "Everyone makes mistakes. Instead of repeatedly blaming "
            "yourself, understand what happened and what you can "
            "learn from it. If you made a mistake, think about one "
            "way to handle the situation better next time."
        ),
        "medium": (
            "If your actions have hurt someone, take responsibility "
            "and offer a sincere apology. Where possible, make an "
            "effort to correct the mistake. At the same time, remember "
            "that learning from an action is more useful than punishing "
            "yourself repeatedly."
        ),
        "high": (
            "It seems that guilt is taking a heavy emotional toll "
            "on you. Accept responsibility for what you can change, "
            "but remember that you are more than one mistake. Talk "
            "to someone you trust or a mental health professional "
            "if these feelings are overwhelming or will not go away."
        )
    },

    "attachment": {
        "low": (
            "Caring deeply about someone or something is natural. "
            "At the same time, make room for your own interests, "
            "responsibilities and relationships. Enjoy what matters "
            "to you without forgetting to take care of yourself."
        ),
        "medium": (
            "If your happiness depends too much on another person "
            "or a particular situation, try rebuilding your own "
            "routine and interests. Give others space, respect their "
            "choices and maintain healthy boundaries. Your personal "
            "well-being also deserves attention."
        ),
        "high": (
            "It sounds like this attachment is causing you significant "
            "distress. Try to reconnect with activities and people "
            "outside this attachment. Set boundaries that protect "
            "your well-being and speak with someone you trust. "
            "Professional support can help if you feel unable to "
            "manage these feelings."
        )
    },

    "failure": {
        "low": (
            "A disappointing result does not define your abilities "
            "or your future. Look at what did not work and identify "
            "one thing you can improve. Give yourself another chance "
            "without comparing your progress with someone else's."
        ),
        "medium": (
            "Repeated setbacks can make anyone lose confidence. "
            "Take a closer look at your approach and identify what "
            "you can change. Ask for honest feedback from someone "
            "experienced and create a more realistic plan. Progress "
            "does not always happen on the first attempt."
        ),
        "high": (
            "I understand that this failure may feel like a major "
            "setback. Please do not make it a judgment about your "
            "entire life. Take time to recover, speak with someone "
            "supportive and consider other possible paths. You can "
            "learn from this experience without allowing it to "
            "define your worth."
        )
    },

    "relationship": {
        "low": (
            "Misunderstandings can happen in any relationship. "
            "Try to understand the other person's perspective "
            "before reacting. Explain your feelings honestly and "
            "give each other an opportunity to communicate."
        ),
        "medium": (
            "If the same problems keep occurring, have an honest "
            "conversation about what needs to change. Listen to "
            "each other, express your expectations clearly and "
            "agree on reasonable boundaries. A healthy relationship "
            "requires effort and respect from everyone involved."
        ),
        "high": (
            "If this relationship is causing serious emotional "
            "distress, take your feelings and well-being seriously. "
            "You do not have to tolerate constant disrespect, "
            "threats or abuse. Seek support from someone you trust "
            "and prioritize your safety. Professional guidance "
            "may also help you decide what to do next."
        )
    },

    "greed": {
        "low": (
            "Wanting a better life is natural. However, take a "
            "moment to consider whether you genuinely need what "
            "you are chasing. Appreciate your current progress "
            "and avoid comparing your life with others."
        ),
        "medium": (
            "If the desire for more money, possessions, recognition "
            "or success is affecting your peace, reconsider your "
            "priorities. Set realistic goals and make sure your "
            "ambition does not damage your relationships, health "
            "or responsibilities."
        ),
        "high": (
            "If the constant desire for more is leading you to "
            "neglect important people or compromise your values, "
            "take a step back. Review your habits, set clear limits "
            "and think about what genuinely makes your life "
            "meaningful. Consider seeking guidance if the behaviour "
            "feels difficult to control."
        )
    },

    "purpose": {
        "low": (
            "You do not need to have your entire future planned "
            "out. Think about what interests you, what you enjoy "
            "doing and what matters to you. Explore new activities "
            "and give yourself time to discover what feels meaningful."
        ),
        "medium": (
            "Feeling unsure about your direction can be frustrating. "
            "Think about your strengths, interests and personal "
            "values. Explore different opportunities and talk "
            "with people who have experience in areas that interest "
            "you. Choose one realistic goal to work towards."
        ),
        "high": (
            "When life feels directionless, even ordinary activities "
            "can feel difficult. Do not pressure yourself to find "
            "one perfect purpose immediately. Focus on basic needs, "
            "meaningful relationships and one manageable goal. "
            "Consider talking to a counsellor or mental health "
            "professional if this feeling is persistent or overwhelming."
        )
    },

    "ego": {
        "low": (
            "Sometimes it is difficult to accept another person's "
            "opinion, especially when you strongly believe you are "
            "right. Try listening without immediately defending "
            "yourself. You can learn something useful even when "
            "you disagree."
        ),
        "medium": (
            "If disagreements are affecting your relationships, "
            "consider whether the need to be right is stopping "
            "you from understanding others. Accept constructive "
            "feedback, acknowledge mistakes and give credit to "
            "people who help you."
        ),
        "high": (
            "If pride or the need to control situations is repeatedly "
            "hurting your relationships, take an honest look at "
            "your behaviour. Listen to the people affected, accept "
            "responsibility and apologize where necessary. Building "
            "mutual respect is more valuable than proving yourself "
            "right every time."
        )
    },

    "decision": {
        "low": (
            "For a small decision, identify what matters most to "
            "you and choose the option that fits your needs. Not "
            "every choice needs endless thinking. Give yourself "
            "permission to make a reasonable decision."
        ),
        "medium": (
            "For an important decision, write down your available "
            "options and compare their benefits, risks and possible "
            "consequences. Consider your personal circumstances "
            "and seek advice from someone experienced. Choose "
            "based on reliable information rather than pressure "
            "from others."
        ),
        "high": (
            "When a decision feels overwhelming, avoid rushing "
            "just to escape the pressure. Identify what must be "
            "decided immediately and what can wait. Gather reliable "
            "information and discuss the situation with someone "
            "qualified or trustworthy. Take enough time to understand "
            "the consequences before choosing."
        )
    },

    "loneliness": {
        "low": (
            "Feeling lonely sometimes is a common human experience. "
            "Reach out to a friend, family member or someone you "
            "feel comfortable with. Even a short conversation "
            "can help you feel more connected."
        ),
        "medium": (
            "If you have been feeling disconnected from others, "
            "try creating opportunities for regular interaction. "
            "Join a group, community activity or shared-interest "
            "event. Reconnect with people you have lost touch with "
            "and allow new friendships to develop naturally."
        ),
        "high": (
            "Deep loneliness can be painful, and you should not "
            "have to manage it alone. Contact someone you trust "
            "and explain how you have been feeling. Look for "
            "regular social support through community groups, "
            "support networks or professional guidance. If this "
            "feeling is affecting your mental health, consider "
            "speaking with a mental health professional."
        )
    }
}



def normalize(text: str) -> str:
    return text.lower().replace("'", "").replace("’", "")


def analyze(sentence: str) -> dict[str, int]:
    sentence = normalize(sentence)
    scores = {}
    total_value_emotions = 0
    for tag, keywords in TAGS.items():
        matched = False
        for kw in keywords:
            if " " in kw:
                # Multi-word phrase: substring match
                if kw in sentence:
                    matched = True
                    break
            else:
                # Single word: whole-word match to avoid false positives
                if re.search(r'\b' + re.escape(kw) + r'\b', sentence):
                    matched = True
                    break
        scores[tag] = int(matched)
        total_value_emotions += int(scores[tag])
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

    # Build solutions with level based on (tag_score / total_value_emotions)
    total = sum(scores.values())
    matched_solutions = {}
    for tag, score in scores.items():
        if score <= 0 or tag not in SOLUTIONS:
            continue

        relative = score / total if total else 0.0

        if relative < LOW_THRESHOLD:
            level = "low"
        elif relative >= HIGH_THRESHOLD:
            level = "high"
        else:
            level = "medium"

        matched_solutions[tag] = {
            "level": level,
            "text": SOLUTIONS[tag][level],
        }

    return scores, prediction, matched_solutions


@app.post("/")
async def submit_content(request: Request, content: str = Form("")):
    scores, prediction, solutions = solve(content)
    return templates.TemplateResponse(
        request=request,
        name="content.html",
        context={
            "content": content,
            "scores": scores,
            "prediction": prediction,
            "solutions": solutions,
        },
    )


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")