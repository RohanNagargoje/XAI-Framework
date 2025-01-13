from lime.lime_tabular import LimeTabularExplainer

class LIMEExplainer:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.explainer = LimeTabularExplainer(X_train, training_labels=y_train, mode='classification')

    def explain(self, X):
        explanation = self.explainer.explain_instance(X[0], self.model.predict)
        explanation.show_in_notebook()
