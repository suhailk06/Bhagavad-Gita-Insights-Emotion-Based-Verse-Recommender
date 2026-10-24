TAGS = {
    "angry":        ["angry", "mad", "furious", "rage", "hate", "irritated", "annoyed", "frustrated"],
    "sad":          ["sad", "cry", "unhappy", "depressed", "down", "miserable", "tears", "hopeless"],
    "fear":         ["fear", "scared", "afraid", "terrified", "panic", "worried", "anxious", "nervous"],
    "confused":     ["confused", "lost", "unclear", "unsure", "puzzled", "dont know", "do not know",
                     "no idea", "what to do", "cant figure"],
    "stressed":     ["stressed", "pressure", "overwhelmed", "burnout", "tension", "too much",
                     "exam", "deadline", "no time", "running out"],
    "demotivated":  ["demotivated", "lazy", "give up", "no energy", "unmotivated", "tired",
                     "cant focus", "procrastinat"],
    "guilt":        ["guilt", "guilty", "sorry", "blame", "regret", "my fault", "should have"],
    "attachment":   ["attachment", "attached", "miss", "cling", "cant let go", "dependent", "obsessed"],
    "failure":      ["failure", "failed", "loser", "not good enough", "useless", "worthless",
                     "didnt start", "did not start", "didnt study", "did not study", "wasted"],
    "relationship": ["relationship", "partner", "boyfriend", "girlfriend", "friend", "family",
                     "love", "breakup", "crush"],
    "greed":        ["greed", "greedy", "want more", "never enough", "selfish", "possessive"],
    "purpose":      ["purpose", "meaning", "why am i", "goal", "direction", "lost in life",
                     "pointless", "no point"],
    "ego":          ["ego", "pride", "arrogant", "superior", "i am right", "selfish"],
    "decision":     ["decision", "decide", "choice", "choose", "dilemma", "what to do",
                     "cant decide", "should i"],
    "loneliness":   ["lonely", "alone", "isolated", "no one", "empty", "left out", "nobody"],
    
}

def normalize(text):
    return text.lower().replace("'", "").replace("’", "")

def analyze(sentence):
    sentence = normalize(sentence)
    scores = {}
    for tag, keywords in TAGS.items():
        count = sum(1 for kw in keywords if kw in sentence)
        scores[tag] = 1 if count>0 else 0

    return scores



print("Type your feelings:\n")


text = input("You: ").strip()
pred=[]
results = analyze(text)
print("Detected tags:")
for tag, score in results.items():
    print(f"  - {tag} (score: {score})")
    pred.append(score)
print(pred)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel("data.xlsx", dtype=str)
X=dataset.iloc[:,8:].values
Y=dataset.iloc[:,4:8].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2)
classifier.fit(X, Y)

pred=classifier.predict(sc.transform([pred]))
print(pred)