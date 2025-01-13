import unittest
from models.decision_tree import DecisionTreeModel
from sklearn.datasets import load_iris

class TestDecisionTreeModel(unittest.TestCase):
    def test_model_training(self):
        model = DecisionTreeModel()
        X_train, X_test, y_train, y_test = model.load_data()
        model.train(X_train, y_train)
        self.assertTrue(model.predict(X_test).shape[0] == y_test.shape[0])

if __name__ == '__main__':
    unittest.main()
