"""
Classifier Evaluation
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassifierEvaluator:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    def accuracy(self):
        return accuracy_score(self.true_labels, self.predicted_labels)

    def precision(self):
        return precision_score(self.true_labels, self.predicted_labels, average='macro')

    def recall(self):
        return recall_score(self.true_labels, self.predicted_labels, average='macro')

    def f1_score(self):
        return f1_score(self.true_labels, self.predicted_labels, average='macro')

    def get_report(self):
        print(f"Accuracy: {self.accuracy()}")
        print(f"Precision: {self.precision()}")
        print(f"Recall: {self.recall()}")
        print(f"F1 score: {self.f1_score()}")