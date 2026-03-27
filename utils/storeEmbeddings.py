import pandas as pd
import numpy as np
import pickle
from Helpers.getEmbeddings import getEmbeddings
import os



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
greece_path = os.path.join(BASE_DIR, "greece")

df = pd.read_csv("greece.csv")

embedder = getEmbeddings(greece_path)
print("^^^^^#### : ",greece_path)

embeddings = []
for path in df["img_path"]:
    embd = embedder.getEmbeddings(path)
    embeddings.append(embd)

embeddings = np.array(embeddings)


with open("../embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)