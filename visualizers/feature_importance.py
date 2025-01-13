import matplotlib.pyplot as plt
import numpy as np

class FeatureImportanceVisualizer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def plot(self):
        importance = self.model.feature_importances_
        indices = np.argsort(importance)

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(len(importance)), importance[indices], align="center")
        plt.yticks(range(len(importance)), [self.feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.show()
