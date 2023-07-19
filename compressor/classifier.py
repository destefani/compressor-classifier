import gzip
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

class TextClassifier:
    def __init__(self, k=2, n_jobs=-1):
        self.k = k
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.training_set = []

    def fit(self, X_train, y_train):
        self.training_set = list(zip(X_train, y_train))

    def compress(self, string):
        return len(gzip.compress(string.encode()))

    def ncd(self, text):
        Cx1 = self.compress(text)
        distances = []
        for (x2, _) in self.training_set:
            Cx2 = self.compress(x2)
            x1x2 = ' '.join([text, x2])
            Cx1x2 = self.compress(x1x2)
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distances.append(ncd)
        sorted_indices = np.argsort(np.array(distances))
        top_k_classes = np.array(self.training_set)[sorted_indices[:self.k], 1]
        unique, counts = np.unique(top_k_classes, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, X_test):
        with Pool(self.n_jobs) as p:
            predictions = list(tqdm(p.imap(self.ncd, X_test), total=len(X_test), desc="Predicting"))
        return predictions
