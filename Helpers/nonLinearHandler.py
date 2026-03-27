import umap

class nonLinear:
    def __init__(self, n_components=50, n_neighbors=15, min_dist=0.1):
        print("Reducer is Called :: ")
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean'
        )

    def fit(self, embeddings):
        self.reducer.fit(embeddings)

    def transform(self, embeddings):
        return self.reducer.transform(embeddings)

    def fit_transform(self, embeddings):
        print("Reducer is Fitted ")
        return self.reducer.fit_transform(embeddings)