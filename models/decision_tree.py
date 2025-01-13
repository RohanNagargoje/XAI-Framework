from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.model_name = "Decision Tree"

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def load_data(self):
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
