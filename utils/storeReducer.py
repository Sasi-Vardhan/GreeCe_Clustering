import os
import numpy as np
import pickle
from Helpers.nonLinearHandler import nonLinear


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
embeddings_path = os.path.join(BASE_DIR, "embeddings.pkl")
umap_model_path = os.path.join(BASE_DIR, "umap_model.pkl")
reduced_embeddings_path = os.path.join(BASE_DIR, "embedding_to_train.pkl")

with open(embeddings_path,"rb") as f:
    data=pickle.load(f)
embeddings=data

reducer=nonLinear()

embedding_to_train=reducer.fit_transform(embeddings)

with open(umap_model_path, "wb") as f:
    pickle.dump(reducer.reducer, f)

with open(reduced_embeddings_path,"wb") as fr:
    pickle.dump(embedding_to_train,fr)

print("Reducer & training Embeddings are stored is Stored")

