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

LOW_THRESHOLD = 0.34
HIGH_THRESHOLD = 0.67


TAGS = {
    "angry": [
        "angry", "anger", "angered", "angrier", "angriest", "mad", "furious",
        "furiously", "rage", "raging", "raged", "hateful", "hating", "hate",
        "irritated", "irritate", "irritates", "irritating", "irritation",
        "annoyed", "annoy", "annoys", "annoying", "annoyance", "frustrated",
        "frustrate", "frustrates", "frustrating", "frustration", "outraged",
        "outrage", "outraging", "bitter", "bitterness", "resentful", "resent",
        "resented", "resenting", "hostile", "hostility", "aggravated",
        "aggravate", "aggravating", "aggravation", "fuming", "seething",
        "infuriated", "infuriate", "infuriating", "livid", "exasperated",
        "exasperate", "exasperating", "displeased", "displease", "displeasure",
        "triggered", "triggering", "enraged", "enrage", "enraging", "wrath",
        "wrathful", "irritable", "irritability", "provoked", "provoking",
        "provocation", "resentment", "aggressive", "aggression", "aggressively",
        "ruthless", "vengeful", "vengeance", "spiteful", "spite", "vindictive",
        "furiousness", "temperamental", "antagonistic", "belligerent",
        "combative", "quarrelsome", "snapped", "shouted", "shouting",
        "yelled", "yelling", "screamed", "screaming", "argued", "arguing"
    ],

    "sad": [
        "sad", "sadder", "saddest", "sadness", "sadden", "saddened",
        "saddening", "cry", "cries", "cried", "crying", "unhappy",
        "unhappier", "unhappiest", "depressed", "depress", "depresses",
        "depressing", "depression", "downcast", "miserable", "misery",
        "tears", "tearful", "hopeless", "hopelessness", "grief", "grieving",
        "grieved", "heartbroken", "heartbreaking", "gloomy", "gloominess",
        "sorrow", "sorrowful", "sorrowing", "upset", "devastated",
        "devastate", "devastating", "glum", "crestfallen", "despair",
        "despairing", "despaired", "melancholy", "weeping", "low",
        "hurting", "hurt", "hurtful", "broken", "emptiness", "empty",
        "lonely", "loneliness", "forlorn", "woeful", "woe", "mourn",
        "mourned", "mourning", "mournful", "lament", "lamented",
        "lamenting", "regretful", "disappointed", "disappointing",
        "disappointment", "discouraged", "discouraging", "discouragement",
        "dejected", "dejection", "downhearted", "heavyhearted", "blue",
        "tearfully", "painful", "painfully", "anguish", "anguished",
        "suffering", "suffered", "suffer", "grieved", "desolate",
        "desolation", "disheartened", "dishearten", "disheartening",
        "heartache", "aching", "ache", "ached", "wistful", "wistfulness"
    ],

    "fear": [
        "fear", "fears", "feared", "fearing", "fearful", "fearfully",
        "fearless", "fearlessness", "scared", "scare", "scares", "scaring",
        "afraid", "terrified", "terrify", "terrifies", "terrifying",
        "terror", "terrorized", "terrorize", "terrifyingly", "panic",
        "panicked", "panicking", "panicky", "worried", "worry", "worries",
        "worrying", "anxious", "anxiety", "anxiously", "nervous",
        "nervously", "nervousness", "dread", "dreads", "dreaded", "dreading",
        "frightened", "frighten", "frightens", "frightening", "jittery",
        "alarmed", "alarm", "alarming", "alarmed", "edgy", "paranoid",
        "paranoia", "petrified", "petrify", "threatened", "threaten",
        "threatening", "uneasy", "uneasiness", "apprehensive", "apprehension",
        "timid", "timidity", "phobia", "phobic", "freaking", "doom",
        "doomed", "foreboding", "forebodingly", "insecure", "insecurity",
        "startled", "startle", "startling", "shaky", "shaking", "tremble",
        "trembled", "trembling", "quivering", "quivered", "quiver",
        "horrified", "horrify", "horrifying", "horror", "horrific",
        "distressed", "distress", "distressing", "hesitant", "hesitancy",
        "flustered", "flustering", "intimidated", "intimidate",
        "intimidating", "intimidation", "alarmed", "apprehensively"
    ],

    "confused": [
        "confused", "confuse", "confuses", "confusing", "confusion",
        "confusingly", "lost", "unclear", "uncertain", "uncertainty",
        "unsure", "unsureness", "puzzled", "puzzle", "puzzles", "puzzling",
        "perplexed", "perplex", "perplexes", "perplexing", "perplexity",
        "baffled", "baffle", "baffles", "baffling", "clueless",
        "disoriented", "disorient", "disorienting", "befuddled", "befuddle",
        "befuddling", "muddled", "muddle", "muddling", "mystified",
        "mystify", "mystifies", "mystifying", "bewildered", "bewilder",
        "bewildering", "bewilderment", "uncertainly", "ambiguous",
        "ambiguity", "ambivalently", "ambivalent", "ambivalence",
        "undecided", "indecisive", "indecision", "hesitant", "hesitate",
        "hesitates", "hesitated", "hesitating", "doubtful", "doubt",
        "doubts", "doubted", "doubting", "questioning", "questioned",
        "uncertainly", "unresolved", "incomprehensible", "incomprehension",
        "incomprehensible", "incomprehensibly", "unintelligible",
        "unintelligibly", "disbelief", "disbelieving", "perplexingly",
        "flustered", "fluster", "flusters", "flustering", "disconcerted",
        "disconcerting", "disconcert", "confound", "confounded",
        "confounding", "obscure", "obscured", "obscurity", "vague",
        "vagueness", "uncertain"
    ],

    "stressed": [
        "stress", "stresses", "stressed", "stressing", "stressful",
        "stressfully", "pressure", "pressured", "pressuring", "pressurize",
        "pressurized", "pressurizing", "overwhelmed", "overwhelm",
        "overwhelms", "overwhelming", "overwhelmingly", "burnout",
        "burned", "burning", "burnt", "tension", "tensions", "tense",
        "tensed", "tensing", "strained", "strain", "strains", "straining",
        "swamped", "swamp", "swamping", "burdened", "burden", "burdens",
        "burdensome", "frazzled", "chaotic", "chaos", "chaotically",
        "drowning", "crushing", "crushed", "crush", "piling", "exhausted",
        "exhaust", "exhausts", "exhausting", "exhaustion", "fatigued",
        "fatigue", "fatiguing", "fatiguingly", "overloaded", "overload",
        "overloading", "overworked", "overwork", "overworking", "hectic",
        "frantic", "frantically", "franticness", "pressing", "demanding",
        "demanded", "demands", "taxing", "taxed", "tax", "taxes",
        "troubled", "trouble", "troubles", "troubling", "agitated",
        "agitate", "agitation", "restless", "restlessness", "frazzling",
        "harried", "harrying", "rushed", "rushing", "hurried", "hurrying",
        "burdening", "overextension", "overextended", "frustrated",
        "juggling", "strained", "struggling", "struggled", "struggle"
    ],

    "demotivated": [
        "demotivated", "demotivate", "demotivates", "demotivating",
        "demotivation", "lazy", "lazier", "laziest", "laziness",
        "unmotivated", "motivationless", "tired", "tire", "tires",
        "tiring", "tiredness", "exhausted", "exhausting", "exhaustion",
        "uninterested", "disinterested", "apathetic", "apathy", "apathetically",
        "sluggish", "sluggishness", "drained", "drain", "drains", "draining",
        "uninspired", "inspire", "inspires", "inspiring", "inspiration",
        "fatigued", "fatigue", "fatiguing", "lethargic", "lethargy",
        "listless", "listlessness", "inert", "inertia", "stagnant",
        "stagnation", "procrastinate", "procrastinates", "procrastinated",
        "procrastinating", "procrastination", "unproductive", "unproductively",
        "unproductivity", "disengaged", "disengage", "disengaging",
        "discouraged", "discourage", "discourages", "discouraging",
        "slumped", "slumping", "slump", "stuck", "unenthusiastic",
        "enthusiasm", "unwilling", "unwillingness", "passive", "passivity",
        "indifferent", "indifference", "weary", "wearier", "weariest",
        "weariness", "listlessly", "unambitious", "aimless", "aimlessness",
        "apathetic", "hesitant", "hesitated", "hesitating", "avoiding",
        "avoided", "avoid", "avoids", "neglecting", "neglected", "neglect",
        "neglectful", "unfocused", "distracted", "distract", "distracting",
        "distraction", "uncommitted", "disheartened", "disheartening"
    ],

    "guilt": [
        "guilt", "guilty", "guiltily", "guiltiness", "sorry", "sorrowful",
        "blame", "blames", "blamed", "blaming", "blameless", "blameworthy",
        "regret", "regrets", "regretted", "regretting", "regretful",
        "regretfully", "ashamed", "shame", "shames", "shamed", "shaming",
        "shameful", "shamefully", "shameless", "shamelessly", "remorse",
        "remorseful", "remorsefully", "conscience", "conscientious",
        "penitent", "penitence", "penitently", "accountable", "accountability",
        "selfblame", "selfblaming", "fault", "faulty", "faultless",
        "faulting", "repent", "repents", "repented", "repenting",
        "repentant", "repentance", "contrite", "contrition", "apologetic",
        "apologetically", "apologize", "apologizes", "apologized",
        "apologizing", "apology", "apologies", "atonement", "reproach",
        "reproached", "reproaching", "selfreproach", "selfreproachful",
        "culpable", "culpability", "culpably", "penance", "penitent",
        "confession", "confessed", "confessing", "confess", "confesses",
        "wrongdoing", "wrongdoer", "transgressed", "transgression",
        "transgress", "transgressing", "remorsefully", "regretfully",
        "repentant", "repentantly", "selfcritical", "selfcriticism",
        "selfcondemnation", "condemned", "condemning", "condemn",
        "condemns", "unforgiven", "unforgivable", "forgiveness",
        "unforgiving"
    ],

    "attachment": [
        "attachment", "attachments", "attached", "attaches", "attaching",
        "attach", "detached", "detach", "detaches", "detaching",
        "detachment", "miss", "misses", "missed", "missing",
        "cling", "clings", "clung", "clinging", "clingy", "clinginess",
        "dependent", "dependence", "dependency", "depending", "depended",
        "obsessed", "obsess", "obsesses", "obsessing", "obsession",
        "obsessional", "fixated", "fixate", "fixates", "fixating",
        "fixation", "needy", "neediness", "codependent", "codependency",
        "crave", "craves", "craved", "craving", "inseparable",
        "hooked", "hook", "hooking", "reliant", "reliance", "rely",
        "relies", "relied", "relying", "possessive", "possessiveness",
        "possess", "possesses", "possessed", "possessing", "stalking",
        "stalk", "stalks", "stalked", "addicted", "addiction", "addictive",
        "longing", "longed", "long", "yearning", "yearn", "yearned",
        "pining", "pined", "pine", "desperate", "desperation",
        "infatuated", "infatuation", "infatuate", "devoted", "devotion",
        "devote", "devotes", "devoting", "dependently", "overattached",
        "overprotective", "overprotectiveness", "jealous", "jealousy",
        "jealously", "covet", "covets", "coveted", "coveting",
        "preoccupied", "preoccupation", "consumed", "consuming",
        "yearned", "longingly", "insecurity", "insecure", "abandonment",
        "abandoned", "abandon", "abandons", "abandoning"
    ],

    "failure": [
        "failure", "failures", "fail", "fails", "failed", "failing",
        "loser", "losers", "losing", "lost", "worthless", "worthlessness",
        "useless", "uselessness", "inadequate", "inadequacy", "inadequately",
        "disappointment", "disappointments", "disappointed", "disappoint",
        "disappoints", "disappointing", "incompetent", "incompetence",
        "incompetently", "flop", "flops", "flopped", "flopping",
        "ruined", "ruin", "ruins", "ruining", "mistake", "mistakes",
        "mistaken", "mistaking", "screwed", "screwup", "mess", "messed",
        "messing", "imposter", "impostor", "imposters", "impostors",
        "insecure", "insecurity", "incapable", "incapability",
        "unqualified", "unsuccessful", "unsuccessfully", "unsuccessful",
        "underachiever", "underachieving", "underachieved", "underachieve",
        "defeated", "defeat", "defeats", "defeating", "defeatist",
        "rejected", "reject", "rejects", "rejecting", "rejection",
        "unaccomplished", "unfulfilled", "unrewarded", "ineffective",
        "ineffectively", "ineffectiveness", "inefficient", "inefficiency",
        "inability", "unable", "incapacitated", "flawed", "flaw",
        "flaws", "flawedness", "mediocre", "mediocrity", "inferior",
        "inferiority", "substandard", "disgraced", "disgrace",
        "disgraceful", "humiliated", "humiliate", "humiliating",
        "humiliation", "embarrassed", "embarrass", "embarrassing",
        "embarrassment", "deficient", "deficiency", "lagging", "lagged",
        "underperformed", "underperform", "underperforming"
    ],

    "relationship": [
        "relationship", "relationships", "partner", "partners", "boyfriend",
        "girlfriend", "friend", "friends", "friendship", "friendships",
        "family", "familial", "love", "loved", "loving", "loves",
        "loveless", "lovelessness", "lovable", "lovingly", "breakup",
        "breakups", "crush", "crushes", "crushed", "crushing", "spouse",
        "spouses", "marriage", "marriages", "married", "marrying",
        "husband", "wife", "dating", "dated", "date", "dates", "ex",
        "romance", "romantic", "romantically", "couple", "couples",
        "fiance", "fiancee", "colleague", "colleagues", "relative",
        "relatives", "toxic", "toxicity", "commitment", "committed",
        "committing", "cheating", "cheated", "cheat", "cheats",
        "heartbreak", "heartbroken", "heartbreaking", "betrayed",
        "betray", "betrays", "betraying", "betrayal", "trust", "trusted",
        "trusting", "trustworthy", "trustless", "distrust", "distrusted",
        "distrusting", "jealous", "jealousy", "possessive", "possessiveness",
        "neglected", "neglect", "neglecting", "neglectful", "ignored",
        "ignore", "ignores", "ignoring", "rejected", "reject", "rejects",
        "rejecting", "rejection", "separated", "separate", "separating",
        "separation", "divorced", "divorce", "divorcing", "divorcee",
        "reconcile", "reconciled", "reconciling", "reconciliation",
        "affection", "affectionate", "affectionately", "intimacy",
        "intimate", "intimately", "bond", "bonded", "bonding", "bonded",
        "disagreement", "disagree", "disagreed", "disagreeing",
        "argument", "arguments", "argue", "argues", "argued", "arguing",
        "misunderstood", "misunderstand", "misunderstands",
        "misunderstanding", "abusive", "abuse", "abused", "abusing",
        "manipulative", "manipulation", "manipulated", "manipulating"
    ],

    "greed": [
        "greed", "greedy", "greedier", "greediest", "greedily",
        "greediness", "selfish", "selfishly", "selfishness", "possessive",
        "possessiveness", "materialistic", "materialism", "covetous",
        "covetously", "covetousness", "covet", "covets", "coveted",
        "coveting", "hoarding", "hoard", "hoards", "hoarded", "grasping",
        "insatiable", "insatiably", "gluttonous", "gluttony", "glutton",
        "selfserving", "stingy", "stingier", "stingiest", "stinginess",
        "powerhungry", "avarice", "avaricious", "avariciously", "acquisitive",
        "acquisitiveness", "rapacious", "rapaciously", "rapacity",
        "voracious", "voraciously", "voracity", "exploitative", "exploiting",
        "exploited", "exploit", "exploits", "entitled", "entitlement",
        "overindulgent", "overindulgence", "indulgent", "indulgence",
        "consuming", "consumed", "consumption", "excessive", "excessively",
        "excess", "extravagant", "extravagance", "extravagantly",
        "lavish", "lavishly", "lavishness", "hoarder", "hoarders",
        "grasped", "grasps", "grasp", "graspingly", "moneyhungry",
        "wealthy", "wealth", "wealthier", "wealthiest", "profitseeking",
        "profiteering", "profiteer", "profiteers", "exploitative",
        "selfcentered", "selfcenteredness", "egocentric", "egocentrism",
        "narrowminded", "uncharitable", "uncharitably", "ungenerous",
        "ungenerously", "miserly", "miserliness", "avariciousness",
        "rapaciousness", "acquisitive", "acquired", "acquiring"
    ],

    "purpose": [
        "purpose", "purposes", "purposeful", "purposefully", "purposeless",
        "purposelessness", "meaning", "meaningful", "meaningfully",
        "meaningless", "meaninglessness", "goal", "goals", "goalless",
        "direction", "directionless", "directionlessnes", "aimless",
        "aimlessly", "aimlessness", "calling", "significance", "significant",
        "insignificant", "insignificance", "emptiness", "empty", "emptied",
        "empties", "emptying", "vacant", "vacancy", "pointless",
        "pointlessness", "aim", "aims", "aimed", "aiming", "adrift",
        "existential", "existence", "existing", "exist", "exists",
        "existed", "existentially", "void", "hollow", "hollowness",
        "fulfillment", "fulfilment", "fulfilled", "fulfilling", "fulfill",
        "fulfills", "fulfil", "fulfils", "unfulfilled", "unfulfilling",
        "unrealized", "unrealised", "realize", "realized", "realizing",
        "realise", "realised", "realising", "aspiration", "aspirations",
        "aspire", "aspires", "aspired", "aspiring", "vision", "visions",
        "visionless", "directional", "misguided", "wandering", "wandered",
        "wander", "wanders", "drifting", "drifted", "drift", "drifts",
        "stagnant", "stagnation", "unmotivated", "unfocused", "lost",
        "identity", "identityless", "meaningfully", "insignificantly",
        "disconnected", "disconnection", "unanchored", "rootless",
        "rootlessness", "unsettled", "aimlessly", "futile", "futility",
        "futilely", "emptyhearted", "disillusioned", "disillusionment"
    ],

    "ego": [
        "ego", "egos", "egoism", "egoist", "egoistic", "egotistical",
        "egotism", "egotist", "egotistically", "pride", "proud", "prouder",
        "proudest", "proudly", "arrogant", "arrogance", "arrogantly",
        "superior", "superiority", "hubris", "selfimportance",
        "conceited", "conceit", "conceitedly", "boastful", "boastfully",
        "boast", "boasts", "boasted", "boasting", "stubborn", "stubbornly",
        "stubbornness", "narcissistic", "narcissism", "narcissist",
        "haughty", "haughtily", "haughtiness", "vain", "vanity", "vainly",
        "knowitall", "selfcentered", "selfcenteredness", "selfabsorbed",
        "selfabsorption", "selfish", "selfishly", "selfishness", "snobbish",
        "snobbery", "snobbishly", "presumptuous", "presumption",
        "presumptuously", "domineering", "domination", "dominate",
        "dominates", "dominated", "dominating", "controlling", "control",
        "controls", "controlled", "controlling", "authoritarian",
        "dictatorial", "bossy", "bossiness", "imperious", "imperiously",
        "overconfident", "overconfidence", "cocky", "cockiness",
        "smug", "smugness", "smugly", "pompous", "pompously", "pomposity",
        "pretentious", "pretentiously", "pretentiousness", "elitist",
        "elitism", "condescending", "condescend", "condescended",
        "condescension", "disdainful", "disdain", "disdainfully",
        "dismissive", "dismissiveness", "intolerant", "intolerance",
        "inflexible", "inflexibility", "opinionated", "dogmatic",
        "dogmatism", "uncompromising", "uncompromisingly", "entitled",
        "entitlement", "selfrighteous", "selfrighteously",
        "selfrighteousness", "imperiousness", "boastfulness"
    ],

    "decision": [
        "decision", "decisions", "decide", "decides", "decided", "deciding",
        "decisive", "decisively", "decisiveness", "indecisive", "indecision",
        "indecisively", "choice", "choices", "choose", "chooses", "chose",
        "chosen", "choosing", "dilemma", "dilemmas", "hesitate", "hesitates",
        "hesitated", "hesitating", "hesitation", "hesitant", "hesitancy",
        "conflicted", "conflict", "conflicts", "conflicting", "ambivalent",
        "ambivalence", "ambivalently", "weighing", "weighed", "weigh",
        "weighs", "options", "optional", "option", "crossroads",
        "uncertain", "uncertainty", "uncertainly", "doubt", "doubts",
        "doubted", "doubting", "questioning", "questioned", "question",
        "questions", "secondguessing", "secondguessed", "paralyzed",
        "paralyze", "paralyzes", "paralyzing", "paralysis", "overthinking",
        "overthought", "overthink", "overthinks", "overanalyzing",
        "overanalyze", "overanalyzed", "overanalyzes", "evaluating",
        "evaluated", "evaluate", "evaluates", "evaluation", "consider",
        "considers", "considered", "considering", "contemplating",
        "contemplated", "contemplate", "contemplates", "deliberate",
        "deliberates", "deliberated", "deliberating", "deliberation",
        "resolving", "resolved", "resolve", "resolves", "resolution",
        "determining", "determined", "determine", "determines",
        "unclear", "torn", "unsure", "undecided", "unresolved",
        "perplexed", "puzzled", "ponder", "ponders", "pondered",
        "pondering", "reflection", "reflect", "reflects", "reflected",
        "reflecting", "prioritize", "prioritizes", "prioritized",
        "prioritizing", "prefer", "prefers", "preferred", "preferring"
    ],

    "loneliness": [
        "lonely", "lonelier", "loneliest", "loneliness", "alone",
        "isolated", "isolate", "isolates", "isolating", "isolation",
        "isolatedly", "empty", "emptiness", "excluded", "exclude",
        "excludes", "excluding", "exclusion", "outcast", "outcasts",
        "alienated", "alienate", "alienates", "alienating", "alienation",
        "friendless", "desolate", "desolation", "unwanted", "outsider",
        "outsiders", "unseen", "forgotten", "forget", "forgets",
        "forgot", "forgetting", "invisible", "invisibility", "abandoned",
        "abandon", "abandons", "abandoning", "abandonment", "detached",
        "detach", "detaches", "detaching", "detachment", "disconnected",
        "disconnect", "disconnects", "disconnecting", "disconnection",
        "unloved", "loveless", "lovelessness", "unnoticed", "ignored",
        "ignore", "ignores", "ignoring", "neglected", "neglect",
        "neglects", "neglecting", "neglectful", "unimportant",
        "unwelcome", "uninvited", "unaccepted", "unappreciated",
        "unsupported", "helpless", "helplessness", "isolating",
        "withdrawn", "withdraw", "withdraws", "withdrew", "withdrawing",
        "rejected", "reject", "rejects", "rejecting", "rejection",
        "friendlessness", "solitary", "solitude", "solitarily",
        "unaccompanied", "forlorn", "forlornness", "deserted",
        "desert", "deserts", "deserting", "forsaken", "forsake",
        "forsakes", "forsaking", "estranged", "estrange", "estrangement",
        "distant", "distance", "distanced", "distancing", "detached",
        "unconnected", "disengaged", "disengage", "disengagement",
        "unbelonging", "belongingless", "isolative", "unfriendly",
        "friendlessness", "unnoticed", "unwantedness"
    ]
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
                if kw in sentence:
                    matched = True
                    break
            else:
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
    Y = dataset.iloc[:, 3:8].astype(str).values
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