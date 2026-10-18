# 📖 Bhagavad Gita Insights - Emotion-Based Verse Recommender

A machine learning project that analyzes your feelings and recommends relevant verses from the Bhagavad Gita to provide guidance and wisdom.

---

## 🌟 Overview

This project uses **emotion detection** combined with **K-Nearest Neighbors (KNN)** classification to match your current emotional state with the most relevant verses from the Bhagavad Gita. Whether you're feeling stressed, confused, lonely, or seeking purpose, the Gita has timeless wisdom to offer.

---

## 🎯 How It Works

1. **Input Your Feelings** — Type how you're feeling in natural language
2. **Emotion Detection** — The system analyzes your text and detects emotional tags
3. **KNN Matching** — Matches your emotional profile against 700 verses from the Gita
4. **Receive Guidance** — Get the most relevant verse (Chapter, Verse, Sanskrit, Hindi, English translations, and a simple explanation)

---

## 📂 Project Structure

```
├── data.xlsx          # Dataset containing 700 Bhagavad Gita verses with emotion tags
├── project.py         # Main script
└── README.md          # Documentation
```

### Dataset Columns (`data.xlsx`)

| Column | Description |
|--------|-------------|
| `S.No.` | Serial number |
| `Title` | Yoga chapter title |
| `Chapter` | Chapter number |
| `Verse` | Verse reference |
| `Sanskrit Anuvad` | Sanskrit text |
| `Hindi Anuvad` | Hindi translation |
| `Enlgish Translation` | English translation |
| `Simple English Explanation` | Simplified meaning |
| `angry`, `sad`, `fear`, `confused`, `stressed`, `demotivated`, `guilt`, `attachment`, `failure`, `relationship`, `greed`, `purpose`, `ego`, `decision`, `loneliness` | Binary emotion tags (0/1) |

---

## 🧠 Emotion Categories (15 Total)

| Emotion | Example Keywords |
|---------|------------------|
| **Angry** | angry, mad, furious, rage, irritated |
| **Sad** | sad, cry, depressed, hopeless |
| **Fear** | scared, afraid, anxious, nervous |
| **Confused** | lost, unsure, puzzled, no idea |
| **Stressed** | overwhelmed, burnout, exam, deadline |
| **Demotivated** | lazy, give up, tired, procrastinating |
| **Guilt** | guilty, regret, my fault, should have |
| **Attachment** | attached, miss, cling, can't let go |
| **Failure** | failed, loser, not good enough |
| **Relationship** | partner, breakup, family, love |
| **Greed** | greedy, want more, never enough |
| **Purpose** | meaning, goal, pointless, lost in life |
| **Ego** | pride, arrogant, superior |
| **Decision** | dilemma, can't decide, should I |
| **Loneliness** | lonely, alone, isolated, nobody |

---

## 🛠️ Requirements

Install the required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data handling |
| `matplotlib` | Visualization (optional) |
| `scikit-learn` | KNN classifier & StandardScaler |
| `openpyxl` | Reading `.xlsx` files |

---

## 🚀 Usage

### 1. Place the files in the same directory
```
project.py
data.xlsx
```

### 2. Run the script
```bash
python project.py
```

### 3. Enter your feelings when prompted
```
Type your feelings:

You: I feel so stressed about my exams and I can't focus
```

### 4. View detected tags and recommended verse
```
Detected tags:
  - stressed (score: 1)
  - demotivated (score: 1)
  - fear (score: 1)
  ...

Recommended Verse:
['Chapter 2', 'Verse 2.47', 'कर्मण्येवाधिकारस्ते...', 'Focus on doing the right thing...']
```

---

## ⚙️ Technical Details

### Pipeline

```
User Input
   ↓
Text Normalization (lowercase, strip apostrophes)
   ↓
Keyword Matching → Binary Emotion Vector (15 dims)
   ↓
StandardScaler (fit on dataset features)
   ↓
KNN Classifier (k=1, Euclidean distance)
   ↓
Predicted Verse (Chapter, Verse, Sanskrit, Hindi, English)
```

### Model Configuration
- **Algorithm:** K-Nearest Neighbors (`KNeighborsClassifier`)
- **k (neighbors):** `1`
- **Metric:** Minkowski (p=2, i.e., Euclidean)
- **Features:** 15 emotion tags
- **Target:** 4 text columns (Chapter, Verse, Sanskrit, Hindi/English)

> ⚠️ **Note:** The dataset is loaded with `dtype=str`, so the emotion tag columns are treated as strings. Consider converting them to integers for improved model behavior.

---

## 🔧 Customization

### Add New Emotions
Edit the `TAGS` dictionary in `project.py`:

```python
TAGS = {
    "angry": [...],
    "hopeful": ["hopeful", "optimistic", "looking forward"],  # new tag
    # ...
}
```
> ⚠️ Also add the corresponding column in `data.xlsx` and re-tag verses.

### Tune the Classifier
```python
classifier = KNeighborsClassifier(
    n_neighbors=3,        # more neighbors = smoother
    metric='minkowski',
    p=2                   # 2 = Euclidean, 1 = Manhattan
)
```

### Try Other Algorithms
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
```

---

## 📌 Known Limitations

- **Keyword-based detection** — Simple substring matching, not semantic understanding (e.g., "I'm not sad" would match "sad").
- **Single-label output** — Only returns one verse; no ranking or top-N list.
- **No handling of negation** — Contractions like "can't" are normalized but negations aren't parsed.
- **Fixed threshold** — Any keyword match produces a `1`, regardless of intensity.

---

## 🔮 Future Improvements

- [ ] Use NLP models (e.g., spaCy, HuggingFace Transformers) for semantic emotion detection
- [ ] Return **Top-5 verses** instead of one
- [ ] Add **negation handling** ("I'm not stressed")
- [ ] Build a **CLI or web interface** (Flask/Streamlit)
- [ ] Add **multilingual input support** (Hindi, Sanskrit)
- [ ] Include **confidence scores** and explanations
- [ ] Fix `dtype` handling so emotion columns are numeric

---

## 🙏 Acknowledgments

- **Dataset:** 700 verses from the Bhagavad Gita (Chapters 1–18)
- **Inspiration:** Timeless wisdom to help navigate modern emotional challenges

---

## 📜 License

This project is for educational purposes. The Bhagavad Gita is a public-domain spiritual text.

---

> *"You have the right to work, but never to the fruit of work."* — **Bhagavad Gita 2.47**
