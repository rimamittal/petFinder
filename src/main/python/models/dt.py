from sklearn.tree import DecisionTreeClassifier

from src.main.python.models.runModels import run_predictions

dt = DecisionTreeClassifier(random_state=1, max_depth=10)
run_predictions(dt, "Decision Tree Classifier")