from sklearn.neural_network import MLPClassifier

from src.main.python.models.runModels import run_predictions

mlp = MLPClassifier()
run_predictions(mlp, 'Preceptron')