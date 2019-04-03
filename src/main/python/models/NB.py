from sklearn.naive_bayes import GaussianNB

# NB
from src.main.python.models.runModels import run_predictions

NB = GaussianNB()
run_predictions(NB, 'Naive Bayes')