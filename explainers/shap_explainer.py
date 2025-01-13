import shap

class SHAPExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, X):
        shap_values = self.explainer.shap_values(X)
        shap.summary_plot(shap_values, X)
