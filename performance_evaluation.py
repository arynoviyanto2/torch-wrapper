from enum import Enum
import numpy as np

class Metric(Enum):
    ACCURACY = 'accuracy'

class PerformanceEvaluation:
    def __init__(self, outputs, labels):
        self.outputs = outputs
        self.labels = labels

    def get_performance_metrics(self):
        return {
            Metric.ACCURACY: self.accuracy()
        }

    # Define performance evaluation methods here
    def accuracy(self):
        predicted_labels = np.argmax(self.outputs, axis=1)
        return np.sum(predicted_labels == self.labels) / float(self.labels.size)