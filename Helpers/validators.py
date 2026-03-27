import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ValidateClusters:
    def __init__(self, embeddings, labels, k, score) -> None:
        self.embeddings = embeddings
        self.labels = labels
        self.k = k
        self.score = score
        
        self.chunk()

    def chunk(self):
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.embeddings,
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels
        )
    
    def analyseLabels(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def validate(self):
        label_dist = self.analyseLabels()

        clf = LogisticRegression(max_iter=1000)
        clf.fit(self.x_train, self.y_train)

        y_pred = clf.predict(self.x_valid)
        accuracy = accuracy_score(self.y_valid, y_pred)

        return {
            "k": self.k,
            "silhouette": self.score,
            "accuracy": accuracy,
            "label_distribution": label_dist,
            "model": clf
        }