# emotion_tagger.py
# Basic keyword-based emotion/topic tagger (v2 - normalized + more phrases)

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
    """Lowercase + remove apostrophes so 'don't' == 'dont'."""
    return text.lower().replace("'", "").replace("’", "")

def analyze(sentence):
    sentence = normalize(sentence)
    scores = {}
    for tag, keywords in TAGS.items():
        count = sum(1 for kw in keywords if kw in sentence)
        if count > 0:
            scores[tag] = count
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

def main():
    print("=== Emotion / Topic Tagger ===")
    print("Type a sentence (or 'quit' to exit)\n")

    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not text:
            continue

        results = analyze(text)
        if results:
            print("Detected tags:")
            for tag, score in results:
                print(f"  - {tag} (score: {score})")
        else:
            print("  No matching tags found.")
        print()

if __name__ == "__main__":
    main()