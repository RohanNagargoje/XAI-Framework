from models.decision_tree import DecisionTreeModel
from explainers.shap_explainer import SHAPExplainer
from visualizers.feature_importance import FeatureImportanceVisualizer

class XAIFramework:
    def __init__(self, model, explainer, visualizer):
        self.model = model
        self.explainer = explainer
        self.visualizer = visualizer

    def run(self, X_train, X_test, y_train, y_test):
        # Train model
        self.model.train(X_train, y_train)

        # Generate explanations
        self.explainer.explain(X_test)

        # Visualize feature importance
        self.visualizer.plot()
