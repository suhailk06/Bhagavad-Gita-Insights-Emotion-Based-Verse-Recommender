import chardet
import pandas as pd
import numpy as np

with open("data.csv", "rb") as f:
    raw = f.read()
print(chardet.detect(raw))
# e.g. {'encoding': 'utf-16', 'confidence': 0.99, ...}

df = pd.read_csv("data.csv", sep=None, engine="python",
                 encoding=chardet.detect(raw)["encoding"])