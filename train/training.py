from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Helpers.validators import ValidateClusters
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
embeddings_path = os.path.join(BASE_DIR, "data/embedding_to_train.pkl")
various_k_means=os.path.join(BASE_DIR, "data.pkl")

results = []

with open(embeddings_path,"rb") as f:
    Z_umap=pickle.load(f)

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)

    labels = kmeans.fit_predict(Z_umap)

    score = silhouette_score(Z_umap, labels)

    validator = ValidateClusters(Z_umap, labels, k, score)
    result = validator.validate()

    
    result["labels"] = labels
    result["kmeans"] = kmeans   
    result["k"] = k
    result["silhouette"] = score

    results.append(result)

with open(various_k_means,"wb") as fr:
    pickle.dump(results,fr)